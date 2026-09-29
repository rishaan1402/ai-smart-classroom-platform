"""
WebSocket endpoint — first-message auth protocol + reconnect replay.

Flow:
  1. Client connects (no query string — prevents token leakage in logs)
  2. Server: {"type": "auth_required"}
  3. Client: {"type": "auth", "token": "...", "last_event_id": "uuid|null"} within 5s
  4. Server validates (single-use Redis token), sends {"type": "auth_ok"}
  5. If last_event_id: replay from Redis cache first, PostgreSQL fallback
  6. No auth within 5s → close 4401

Per-IP DoS guard: INCR ws_pending:{ip} TTL 10s; > 5 → close 4429
"""

import asyncio
import json
import logging
import os
from typing import Optional

import redis as redis_lib
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from ..auth import validate_ws_token
from ..database import ApiSession
from ..models import Event

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


def _redis() -> redis_lib.Redis:
    return redis_lib.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
    )


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    client_ip = websocket.client.host if websocket.client else "unknown"

    # Per-IP DoS guard: limit unauthenticated connections
    r = _redis()
    pending_key = f"ws_pending:{client_ip}"
    pending = r.incr(pending_key)
    if pending == 1:
        r.expire(pending_key, 10)
    if pending > 5:
        await websocket.close(code=4429, reason="too many unauthenticated connections")
        return

    await websocket.accept()

    try:
        # 1. Send auth_required
        await websocket.send_json({"type": "auth_required"})

        # 2. Wait up to 5s for auth message
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        except asyncio.TimeoutError:
            await websocket.close(code=4401, reason="auth timeout")
            return

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.close(code=4400, reason="invalid json")
            return

        if msg.get("type") != "auth" or not msg.get("token"):
            await websocket.close(code=4401, reason="missing auth")
            return

        # 3. Validate single-use token
        payload = validate_ws_token(r, msg["token"])
        if not payload:
            await websocket.close(code=4401, reason="invalid token")
            return

        tenant_id = payload.tenant_id
        last_event_id = msg.get("last_event_id")

        # Decrement pending counter on successful auth
        r.decr(pending_key)

        await websocket.send_json({"type": "auth_ok", "tenant_id": tenant_id})

        # 4. Reconnect replay — Redis cache first
        if last_event_id:
            await _replay_events(websocket, tenant_id, last_event_id, r)

        # 5. Live stream via Redis pub/sub
        pubsub = r.pubsub()
        pubsub.subscribe(f"ws_events:{tenant_id}")

        while True:
            try:
                message = pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
                if message and message.get("type") == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()
                    await websocket.send_text(data)
                await asyncio.sleep(0.1)
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS error: {e}")
    finally:
        try:
            r.decr(pending_key)
        except Exception:
            pass


async def _replay_events(
    websocket: WebSocket,
    tenant_id: str,
    last_event_id: str,
    redis_client: redis_lib.Redis,
) -> None:
    """
    Replay events newer than last_event_id.
    Check Redis cache first (recent_events:{tenant_id}), fall back to PostgreSQL.
    """
    max_replay = 100
    try:
        cached = redis_client.lrange(f"recent_events:{tenant_id}", 0, max_replay - 1)
        found_in_cache = False
        to_send = []

        for entry in cached:
            if isinstance(entry, bytes):
                entry = entry.decode()
            try:
                e = json.loads(entry)
                if e.get("id") == last_event_id:
                    found_in_cache = True
                    break
                to_send.append(e)
            except json.JSONDecodeError:
                continue

        if found_in_cache:
            for e in reversed(to_send):  # oldest first
                await websocket.send_json({"type": "replay", "event": e})
            return

        # Not in cache → PostgreSQL fallback with tenant scoping assertion
        db = ApiSession()
        try:
            from sqlalchemy import text
            db.execute(text("SET LOCAL app.tenant_id = :tid"), {"tid": tenant_id})
            target = db.execute(
                select(Event).where(Event.id == last_event_id)
            ).scalar_one_or_none()

            if target:
                rows = db.execute(
                    select(Event)
                    .where(Event.created_at > target.created_at)
                    .order_by(Event.created_at.asc())
                    .limit(max_replay)
                ).scalars().all()

                for e in rows:
                    # Defense in depth: verify tenant match
                    assert str(e.tenant_id) == tenant_id
                    await websocket.send_json({
                        "type": "replay",
                        "event": {
                            "id": str(e.id),
                            "event_type": e.event_type,
                            "severity": e.severity,
                            "created_at": e.created_at.isoformat(),
                        },
                    })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Replay error: {e}")
