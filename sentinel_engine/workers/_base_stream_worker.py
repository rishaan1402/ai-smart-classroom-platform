"""
Base class for Redis Stream consumer workers.

Handles:
- Consumer group auto-creation with mkstream=True
- Jittered reconnect after Redis restart (prevents thundering herd)
- Structured logging with trace_id
- XACK on success, DLQ write on failure
"""

import json
import logging
import random
import threading
import time
from typing import Any, Optional

import redis as redis_lib

logger = logging.getLogger(__name__)


class BaseStreamWorker(threading.Thread):
    """
    Base class for Redis Stream XREADGROUP consumer workers.

    Subclasses implement _process_message(message_id, data).
    """

    def __init__(
        self,
        name: str,
        stream_key: str,
        group_name: str,
        consumer_name: str,
        redis_client: redis_lib.Redis,
        cfg: dict[str, Any],
        dlq_stream: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._stream_key = stream_key
        self._group_name = group_name
        self._consumer_name = consumer_name
        self._redis = redis_client
        self._cfg = cfg
        self._dlq_stream = dlq_stream
        self._stop_event = threading.Event()
        self._tenant_id = cfg.get("tenant_id", "default")

    def run(self) -> None:
        self._ensure_consumer_group()
        logger.info(f"{self.name} started on {self._stream_key}")

        first_read_after_reconnect = False

        while not self._stop_event.is_set():
            try:
                count = 10 if first_read_after_reconnect else 5
                first_read_after_reconnect = False

                messages = self._redis.xreadgroup(
                    groupname=self._group_name,
                    consumername=self._consumer_name,
                    streams={self._stream_key: ">"},
                    count=count,
                    block=1000,  # 1 second block
                )

                if not messages:
                    continue

                for stream_name, entries in messages:
                    for message_id, data in entries:
                        self._handle_message(message_id, data)

            except redis_lib.ConnectionError as e:
                logger.error(f"{self.name} Redis connection lost: {e} — reconnecting with jitter")
                delay = random.uniform(0.5, 5.0)
                self._stop_event.wait(timeout=delay)
                self._ensure_consumer_group()
                first_read_after_reconnect = True

            except redis_lib.RedisError as e:
                logger.error(f"{self.name} Redis error: {e}")
                self._stop_event.wait(timeout=1.0)

        logger.info(f"{self.name} stopped")

    def stop(self) -> None:
        self._stop_event.set()

    def _handle_message(self, message_id: bytes, data: dict) -> None:
        """Handle a single message: process, XACK on success, DLQ on failure."""
        trace_id = data.get(b"trace_id", b"unknown").decode()
        try:
            self._process_message(message_id, data)
            self._redis.xack(self._stream_key, self._group_name, message_id)
            logger.debug(
                f"{self.name} processed message {message_id}",
                extra={"trace_id": trace_id},
            )
        except Exception as e:
            logger.error(
                f"{self.name} failed to process message {message_id}: {e}",
                extra={"trace_id": trace_id},
            )
            self._redis.xack(self._stream_key, self._group_name, message_id)
            self._write_dlq(message_id, data, str(e))

    def _write_dlq(self, message_id: bytes, data: dict, error: str) -> None:
        """Write failed message to DLQ stream."""
        if self._dlq_stream is None:
            return
        try:
            self._redis.xadd(
                self._dlq_stream,
                {
                    "original_stream": self._stream_key,
                    "original_id": message_id,
                    "worker": self.name,
                    "error": error,
                    "timestamp": str(time.time()),
                    "data": json.dumps({k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in data.items()}),
                },
            )
        except Exception as dlq_err:
            logger.error(f"{self.name} failed to write to DLQ: {dlq_err}")

    def _ensure_consumer_group(self) -> None:
        """Create consumer group and stream if they don't exist."""
        try:
            self._redis.xgroup_create(
                self._stream_key, self._group_name, id="0", mkstream=True
            )
        except redis_lib.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def _process_message(self, message_id: bytes, data: dict) -> None:
        """Override in subclasses to process a single message."""
        raise NotImplementedError
