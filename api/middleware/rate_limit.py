"""
Rate limiting via slowapi + Redis.

Limits:
- 100 req/s reads per tenant
- 20 req/s writes per tenant
- /api/auth/ws-token: 5 req/min per user
"""

import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _key_func(request: Request) -> str:
    """Rate limit key: tenant_id from request state, fallback to IP."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id:
        return f"tenant:{tenant_id}"
    return get_remote_address(request)


REDIS_URL = os.getenv("RATE_LIMIT_REDIS_URL", "redis://localhost:6379/1")

limiter = Limiter(
    key_func=_key_func,
    storage_uri=REDIS_URL,
    default_limits=["100/second"],
)
