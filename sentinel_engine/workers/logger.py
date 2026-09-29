"""
EventLogger — Redis Streams consumer that persists events.

Key design decisions:
- NDJSON append-only log: O(1) write per event, crash-safe
  (existing code rewrote entire JSON on every event = O(N))
- velocity_history uses collections.deque(maxlen=30)
  (existing code used list.pop(0) = O(n))
- Writes schema_version, face_recognition_was_active, trace_id on every event
- Daily log rotation via TimedRotatingFileHandler pattern
- Saves JPEG snapshots to storage backend
"""

import json
import logging
import logging.handlers
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import redis as redis_lib

from ._base_stream_worker import BaseStreamWorker
from ..storage import StorageBackend

logger = logging.getLogger(__name__)


class EventLogger(BaseStreamWorker):
    """
    Consumes sentinel:log:{tenant_id} and writes events to:
    1. NDJSON append-only log file (with daily rotation)
    2. PostgreSQL events table (async, via asyncpg or direct write)
    3. MinIO/local storage for snapshot images
    """

    def __init__(
        self,
        redis_client: redis_lib.Redis,
        cfg: dict[str, Any],
        storage: Optional[StorageBackend] = None,
        worker_index: int = 0,
        stream_key: Optional[str] = None,
    ) -> None:
        tenant_id = cfg.get("tenant_id", "default")
        resolved_stream = stream_key or f"sentinel:log:{tenant_id}"
        super().__init__(
            name=f"EventLogger-{worker_index}",
            stream_key=resolved_stream,
            group_name="log_workers",
            consumer_name=f"log_worker_{worker_index}",
            redis_client=redis_client,
            cfg=cfg,
            dlq_stream=f"sentinel:dlq:failed:{tenant_id}",
        )
        self._storage = storage
        self._log_dir = Path(cfg.get("log_dir", "data/logs"))
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # NDJSON append-only log file with daily rotation
        self._ndjson_path = self._log_dir / "events.ndjson"
        self._ndjson_file = self._open_ndjson()
        self._last_rotation_date = self._today()

        # PostgreSQL connection (optional — graceful degradation if unavailable)
        self._db_session = self._init_db_session()

        # Recent events list for WS replay cache (maintained in Redis separately
        # by calling _update_replay_cache)

    def _init_db_session(self):
        """Connect to PostgreSQL and cache tenant/camera UUIDs. Returns None if unavailable."""
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            return None
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker
            engine = create_engine(db_url, pool_size=1, max_overflow=0, pool_pre_ping=True)
            Session = sessionmaker(bind=engine)
            session = Session()
            session.execute(text("SELECT 1"))

            # Cache tenant UUID
            row = session.execute(text("SELECT id FROM tenants LIMIT 1")).fetchone()
            self._tenant_uuid = str(row[0]) if row else None

            # Cache camera name→UUID map for resolving string IDs
            rows = session.execute(text("SELECT id, name, path FROM cameras")).fetchall()
            self._camera_uuid_map: dict[str, str] = {}
            for r in rows:
                cam_uuid = str(r[0])
                self._camera_uuid_map[r[1]] = cam_uuid   # by name
                self._camera_uuid_map[r[2]] = cam_uuid   # by path (e.g. webcam://0)
                # Also map last segment of name for loose matching
                self._camera_uuid_map[r[1].lower()] = cam_uuid

            logger.info(f"EventLogger: PostgreSQL connected, {len(rows)} cameras mapped")
            return session
        except Exception as e:
            logger.warning(f"EventLogger: PostgreSQL unavailable ({e}) — NDJSON only")
            self._tenant_uuid = None
            self._camera_uuid_map = {}
            return None

    def _write_to_postgres(self, event: dict, trace_id: str) -> None:
        """Insert event into PostgreSQL events table."""
        if self._db_session is None:
            return
        try:
            from sqlalchemy import text

            tenant_uuid = getattr(self, "_tenant_uuid", None)
            if tenant_uuid is None:
                return

            # Resolve camera_id string → UUID via cached map
            cam_id_str = event.get("camera_id")
            cam_uuid = None
            if cam_id_str:
                cam_map = getattr(self, "_camera_uuid_map", {})
                cam_uuid = cam_map.get(cam_id_str) or cam_map.get(str(cam_id_str).lower())

            # Strip inline image before storing metadata (saves DB space)
            metadata = dict(event.get("metadata") or {})
            metadata.pop("image_b64", None)

            # Merge threat_description into metadata (not a separate DB column)
            if event.get("threat_description"):
                metadata["threat_description"] = event["threat_description"]

            self._db_session.execute(
                text("""
                    INSERT INTO events (
                        id, schema_version, trace_id, tenant_id, camera_id,
                        event_type, severity, person_id,
                        face_recognition_was_active, enrichment_skipped,
                        threat_level, snapshot_path, clip_path, metadata
                    ) VALUES (
                        :id, :schema_version, :trace_id, :tenant_id, :camera_id,
                        :event_type, :severity, :person_id,
                        :face_recognition_was_active, :enrichment_skipped,
                        :threat_level, :snapshot_path, :clip_path, CAST(:metadata AS jsonb)
                    )
                    ON CONFLICT (id) DO NOTHING
                """),
                {
                    "id": str(uuid.uuid4()),
                    "schema_version": event.get("schema_version", 1),
                    "trace_id": trace_id,
                    "tenant_id": tenant_uuid,
                    "camera_id": cam_uuid,
                    "event_type": event.get("event_type", "UNKNOWN"),
                    "severity": (event.get("severity") or "low").lower(),
                    "person_id": event.get("person_id"),
                    "face_recognition_was_active": event.get("face_recognition_was_active", False),
                    "enrichment_skipped": event.get("enrichment_skipped", False),
                    "threat_level": event.get("threat_level"),
                    "snapshot_path": event.get("snapshot_path"),
                    "clip_path": event.get("clip_path"),
                    "metadata": json.dumps(metadata),
                }
            )
            self._db_session.commit()
        except Exception as e:
            logger.error(f"EventLogger: PostgreSQL write failed: {e}")
            try:
                self._db_session.rollback()
            except Exception:
                pass

    def _process_message(self, message_id: bytes, data: dict) -> None:
        event_json = data.get(b"event", b"{}").decode()
        trace_id = data.get(b"trace_id", b"unknown").decode()

        event = json.loads(event_json)

        # Ensure required fields are present
        event.setdefault("schema_version", 1)
        event.setdefault("trace_id", trace_id)
        event.setdefault("face_recognition_was_active", False)
        event.setdefault("enrichment_skipped", False)

        # Save snapshot if image data present
        img_b64 = event.get("metadata", {}).get("image_b64")
        if img_b64 and self._storage:
            snapshot_path = self._save_snapshot(img_b64, event)
            if snapshot_path:
                event["snapshot_path"] = snapshot_path
                # Remove inline image from log (saves space)
                event.get("metadata", {}).pop("image_b64", None)

        # Rotate log if day changed
        today = self._today()
        if today != self._last_rotation_date:
            self._rotate_log()
            self._last_rotation_date = today

        # Write NDJSON line — O(1) append
        try:
            self._ndjson_file.write(json.dumps(event) + "\n")
            self._ndjson_file.flush()
        except Exception as e:
            logger.error(f"NDJSON write failed: {e}", extra={"trace_id": trace_id})

        # Update WS replay cache in Redis (last 100 events per tenant)
        self._update_replay_cache(event)

        # Persist to PostgreSQL
        self._write_to_postgres(event, trace_id)

        logger.info(
            f"Logged event: {event.get('event_type')} severity={event.get('severity')}",
            extra={"trace_id": trace_id},
        )

    def _save_snapshot(self, img_b64: str, event: dict) -> Optional[str]:
        """Save base64 JPEG snapshot to storage and return path."""
        try:
            import base64
            img_bytes = base64.b64decode(img_b64)
            event_type = event.get("event_type", "unknown")
            person_id = event.get("person_id", 0)
            tenant_id = self._tenant_id
            ts = int(time.time() * 1000)
            filename = f"alert_{person_id}_{event_type}_{ts}.jpg"
            key = f"{tenant_id}/snapshots/{filename}"
            path = self._storage.save(key, img_bytes)
            return path
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def _update_replay_cache(self, event: dict) -> None:
        """Maintain last-100 events per tenant in Redis for WS reconnect replay."""
        try:
            key = f"recent_events:{self._tenant_id}"
            self._redis.lpush(key, json.dumps(event))
            self._redis.ltrim(key, 0, 99)
        except Exception as e:
            logger.debug(f"Failed to update replay cache: {e}")

    def _open_ndjson(self):
        """Open NDJSON file in append mode."""
        return open(self._ndjson_path, "a", encoding="utf-8")

    def _rotate_log(self) -> None:
        """Rename current log and start fresh."""
        try:
            self._ndjson_file.close()
        except Exception:
            pass
        rotated = self._log_dir / f"events.{self._last_rotation_date}.ndjson"
        if self._ndjson_path.exists():
            self._ndjson_path.rename(rotated)
            logger.info(f"Rotated log to {rotated}")
        self._ndjson_file = self._open_ndjson()

    @staticmethod
    def _today() -> str:
        from datetime import date
        return date.today().isoformat()

    def stop(self) -> None:
        super().stop()
        try:
            self._ndjson_file.close()
        except Exception:
            pass
        if self._db_session:
            try:
                self._db_session.close()
            except Exception:
                pass
