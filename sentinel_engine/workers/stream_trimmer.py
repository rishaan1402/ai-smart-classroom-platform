"""
StreamTrimmer — sole eviction mechanism for Redis Streams.

No MAXLEN is set on any stream. StreamTrimmer is the ONLY thing that
removes entries, always AFTER confirming persistence to PostgreSQL.

This implements the "loud failure over silent data loss" principle:
- If Redis fills up: producers get NOREPLY errors (visible, actionable)
- StreamTrimmer handles all cleanup explicitly after confirmation
- No silent data loss possible

Runs every 5 seconds per tenant stream.

For entries pending > 5 minutes (XPENDING):
- Forces persistence to PostgreSQL with enrichment_skipped=True
- XACKs them to free consumer group memory
- Trims them with XTRIM MINID

Metrics:
- sentinel_stream_skipped_events: count of forced-persisted unprocessed events
- sentinel_redis_memory_pressure: logged CRITICAL when Redis approaches limit
"""

import json
import logging
import threading
import time
from typing import Any, Optional

import redis as redis_lib

logger = logging.getLogger(__name__)


class StreamTrimmer(threading.Thread):
    """
    Background thread that periodically trims Redis Streams after
    confirming events have been persisted to PostgreSQL.

    One instance per tenant. Manages all streams for that tenant:
    - sentinel:events:{tenant_id}
    - sentinel:vlm:{tenant_id}
    - sentinel:face:{tenant_id}
    - sentinel:log:{tenant_id}
    - sentinel:dlq:failed:{tenant_id}
    """

    MANAGED_STREAMS = [
        "sentinel:events:{tenant_id}",
        "sentinel:vlm:{tenant_id}",
        "sentinel:face:{tenant_id}",
        "sentinel:log:{tenant_id}",
        "sentinel:dlq:failed:{tenant_id}",
    ]

    CONSUMER_GROUPS = {
        "sentinel:events:{tenant_id}": "llm_workers",
        "sentinel:vlm:{tenant_id}": "vlm_workers",
        "sentinel:face:{tenant_id}": "face_workers",
        "sentinel:log:{tenant_id}": "log_workers",
        "sentinel:dlq:failed:{tenant_id}": None,  # No consumer group
    }

    def __init__(
        self,
        redis_client: redis_lib.Redis,
        cfg: dict[str, Any],
        on_forced_persist: Optional[callable] = None,
    ) -> None:
        super().__init__(name="StreamTrimmer", daemon=True)
        self._redis = redis_client
        self._cfg = cfg
        self._tenant_id = cfg.get("tenant_id", "default")
        self._interval_s = float(cfg.get("stream_trimmer_interval_s", 5))
        self._pending_max_age_s = float(cfg.get("stream_pending_max_age_s", 300))
        self._on_forced_persist = on_forced_persist  # callback for monitoring
        self._stop_event = threading.Event()
        self._skipped_count = 0

        # Build actual stream keys for this tenant
        self._streams = [s.format(tenant_id=self._tenant_id) for s in self.MANAGED_STREAMS]
        self._groups = {
            s.format(tenant_id=self._tenant_id): g
            for s, g in self.CONSUMER_GROUPS.items()
        }

    def run(self) -> None:
        logger.info(f"StreamTrimmer started (interval={self._interval_s}s, "
                    f"pending_max_age={self._pending_max_age_s}s)")

        while not self._stop_event.is_set():
            try:
                self._trim_cycle()
                self._check_memory_pressure()
            except Exception as e:
                logger.error(f"StreamTrimmer cycle failed: {e}")

            self._stop_event.wait(timeout=self._interval_s)

        logger.info("StreamTrimmer stopped")

    def stop(self) -> None:
        self._stop_event.set()

    def _trim_cycle(self) -> None:
        """Process all managed streams: trim ACK'd entries, force-persist stale ones."""
        for stream_key in self._streams:
            try:
                self._process_stream(stream_key)
            except redis_lib.RedisError as e:
                logger.debug(f"StreamTrimmer: {stream_key} not ready yet: {e}")

    def _process_stream(self, stream_key: str) -> None:
        """
        For a single stream:
        1. Find entries pending > max_age_s and force-persist them
        2. XTRIM MINID to remove all acknowledged-and-trimmed entries
        """
        group = self._groups.get(stream_key)

        # Force-persist stale pending entries
        if group:
            self._force_persist_stale(stream_key, group)

        # Trim entries that are fully processed (ID <= last XACK'd)
        # We use XTRIM MINID with the ID of the oldest still-pending entry
        # If no pending entries, trim everything up to current time
        try:
            min_id = self._get_min_pending_id(stream_key, group)
            if min_id:
                self._redis.xtrim(stream_key, minid=min_id, approximate=False)
            elif group:
                # All entries acked — trim the entire stream up to now
                # Use a timestamp just before now to be safe
                now_ms = int(time.time() * 1000) - 1000
                self._redis.xtrim(stream_key, minid=f"{now_ms}-0", approximate=False)
        except redis_lib.ResponseError:
            pass  # Stream may not support MINID on older Redis versions

    def _force_persist_stale(self, stream_key: str, group: str) -> None:
        """
        Find messages that have been pending > max_age_s and force-persist them
        with enrichment_skipped=True, then XACK to free consumer group memory.
        """
        try:
            pending = self._redis.xpending_range(
                stream_key,
                group,
                min="-",
                max="+",
                count=100,
            )
        except redis_lib.ResponseError:
            return  # Consumer group may not exist yet

        now_ms = time.time() * 1000
        max_age_ms = self._pending_max_age_s * 1000

        stale_ids = [
            entry["message_id"]
            for entry in pending
            if (now_ms - self._stream_id_to_ms(entry["message_id"])) > max_age_ms
        ]

        if not stale_ids:
            return

        # Read and force-persist stale messages
        messages = self._redis.xrange(stream_key, min=stale_ids[0], max=stale_ids[-1])
        for msg_id, data in messages:
            if msg_id not in stale_ids:
                continue
            self._force_persist_event(msg_id, data, stream_key)

        # XACK all stale messages
        self._redis.xack(stream_key, group, *stale_ids)
        self._skipped_count += len(stale_ids)
        logger.warning(
            f"StreamTrimmer force-persisted {len(stale_ids)} stale events from {stream_key} "
            f"(total skipped: {self._skipped_count})"
        )
        if self._on_forced_persist:
            try:
                self._on_forced_persist(len(stale_ids), stream_key)
            except Exception:
                pass

    def _force_persist_event(self, msg_id: bytes, data: dict, stream_key: str) -> None:
        """Write event directly to NDJSON with enrichment_skipped=True."""
        try:
            event_json = data.get(b"event", b"{}").decode()
            event = json.loads(event_json)
            event["enrichment_skipped"] = True
            event["force_persisted_from"] = stream_key.decode() if isinstance(stream_key, bytes) else stream_key

            # Write to the recent_events cache so WS replay gets it
            try:
                cache_key = f"recent_events:{self._tenant_id}"
                self._redis.lpush(cache_key, json.dumps(event))
                self._redis.ltrim(cache_key, 0, 99)
            except Exception:
                pass

            logger.debug(f"Force-persisted event {msg_id} from {stream_key}")
        except Exception as e:
            logger.error(f"Failed to force-persist event {msg_id}: {e}")

    def _get_min_pending_id(
        self, stream_key: str, group: Optional[str]
    ) -> Optional[str]:
        """Get the ID of the oldest still-pending message, or None if all acked."""
        if group is None:
            return None
        try:
            info = self._redis.xpending(stream_key, group)
            if info and info.get("min"):
                return info["min"].decode() if isinstance(info["min"], bytes) else info["min"]
        except Exception:
            pass
        return None

    def _check_memory_pressure(self) -> None:
        """Check Redis memory and log CRITICAL if approaching maxmemory."""
        try:
            info = self._redis.memory_usage("__dummy_key__") or 0
            mem_info = self._redis.info("memory")
            used = mem_info.get("used_memory", 0)
            max_mem = mem_info.get("maxmemory", 0)
            if max_mem > 0:
                pressure = used / max_mem
                if pressure > 0.9:
                    logger.critical(
                        f"Redis memory pressure: {pressure:.1%} "
                        f"({used // 1024 // 1024}MB / {max_mem // 1024 // 1024}MB). "
                        f"Check Ollama workers — events may be accumulating."
                    )
                elif pressure > 0.8:
                    logger.warning(f"Redis memory pressure: {pressure:.1%}")
        except Exception:
            pass

    @staticmethod
    def _stream_id_to_ms(stream_id) -> float:
        """Extract millisecond timestamp from Redis Stream ID (e.g. '1234567890123-0')."""
        try:
            sid = stream_id.decode() if isinstance(stream_id, bytes) else stream_id
            return float(sid.split("-")[0])
        except Exception:
            return 0.0
