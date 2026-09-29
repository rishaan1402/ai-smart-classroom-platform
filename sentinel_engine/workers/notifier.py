"""
NotificationDeduplicator — prevents alert fatigue via atomic Redis rate limiting.

Uses SET NX EX (atomic) to prevent TOCTOU race conditions.
(Existing plan had EXISTS then SETEX which allowed duplicate notifications.)

Default: one notification per (tenant, camera, event_type) per 5 minutes.
"""

import logging
from typing import Any

import redis as redis_lib

logger = logging.getLogger(__name__)


class NotificationDeduplicator:
    """
    Rate-limits notifications using Redis atomic SET NX EX.

    key: notif:{tenant_id}:{camera_id}:{event_type}
    TTL: configurable cooldown (default 5 minutes)
    """

    def __init__(self, redis_client: redis_lib.Redis, cfg: dict[str, Any]) -> None:
        self._redis = redis_client
        self._cooldown_s = int(cfg.get("notification_cooldown_s", 300))  # 5 min

    def should_notify(
        self,
        tenant_id: str,
        camera_id: str,
        event_type: str,
    ) -> bool:
        """
        Return True if notification should be sent (first in cooldown window).

        Uses SET NX EX — atomic, no TOCTOU race possible.
        """
        key = f"notif:{tenant_id}:{camera_id}:{event_type}"
        result = self._redis.set(key, 1, nx=True, ex=self._cooldown_s)
        if result:
            logger.debug(f"Notification allowed: {key}")
        else:
            logger.debug(f"Notification suppressed (in cooldown): {key}")
        return result is not None
