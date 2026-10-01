"""
FastAPI application entry point.

Wires middleware, routes, and auto-provisions a default tenant on first startup.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import select

from .database import ApiSession
from .middleware.rate_limit import limiter
from .middleware.tenant import TenantMiddleware
from .models import Tenant
from .routes import auth_routes, cameras, events, health, labeling, lessons, onvif, playback, ws

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SentinelCore API",
    description="AI surveillance platform REST + WebSocket API",
    version="1.0.0",
)

# CORS — dashboard origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tenant middleware
app.add_middleware(TenantMiddleware)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routes
app.include_router(health.router)
app.include_router(auth_routes.router)
app.include_router(events.router)
app.include_router(cameras.router)
app.include_router(onvif.router)
app.include_router(playback.router)
app.include_router(ws.router)
app.include_router(lessons.router)
app.include_router(labeling.router)


@app.on_event("startup")
def auto_provision_default_tenant() -> None:
    """On first startup, create a default tenant if tenants table is empty."""
    db = ApiSession()
    try:
        existing = db.execute(select(Tenant).limit(1)).scalar_one_or_none()
        if not existing:
            tenant = Tenant(name="Default")
            db.add(tenant)
            db.commit()
            logger.info(f"Auto-provisioned default tenant: {tenant.id}")

            env_path = ".env"
            if os.path.exists(env_path):
                with open(env_path, "a") as f:
                    f.write(f"\nDEFAULT_TENANT_ID={tenant.id}\n")
    except Exception as e:
        logger.error(f"Tenant provisioning failed: {e}")
        db.rollback()
    finally:
        db.close()
