"""
Tenant middleware — extracts tenant_id from JWT and sets it on request.state.

This runs before route handlers so that get_db() can use request.state.tenant_id
for SET LOCAL app.tenant_id (RLS).
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from ..auth import JWT_ALGORITHM, JWT_SECRET

# Public paths that don't require authentication
PUBLIC_PATHS = {"/api/health", "/api/auth/login", "/api/auth/register", "/docs", "/openapi.json"}


class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/api/auth/"):
            return await call_next(request)

        # JWT parsing is done in auth dependency; middleware just passes through
        # The actual tenant_id extraction happens in get_current_user()
        return await call_next(request)
