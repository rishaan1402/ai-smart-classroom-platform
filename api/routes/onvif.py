"""
ONVIF camera discovery API endpoint.

POST /api/cameras/onvif/discover
  → Sends WS-Discovery multicast probe, returns found cameras
  → Admin only (network scanning is privileged)

POST /api/cameras/onvif/import
  → Takes a discovered camera and adds it to the DB
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from ..auth import TokenPayload, require_role
from ..deps import get_db
from ..models import Camera

router = APIRouter(prefix="/api/cameras/onvif", tags=["onvif"])


class DiscoverRequest(BaseModel):
    timeout: float = 5.0
    username: str = ""
    password: str = ""


class ImportRequest(BaseModel):
    name: str
    host: str
    path: str
    rtsp_url: Optional[str] = None
    username: str = ""
    password: str = ""


@router.post("/discover")
async def discover(
    body: DiscoverRequest,
    user: TokenPayload = Depends(require_role("admin")),
):
    """
    Probe the local network for ONVIF cameras.
    Returns discovered devices with their RTSP URLs where available.
    """
    from sentinel_engine.onvif_discovery import discover_onvif_cameras

    try:
        cameras = await discover_onvif_cameras(
            timeout=body.timeout,
            username=body.username,
            password=body.password,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Discovery failed: {e}",
        )

    return {
        "found": len(cameras),
        "cameras": cameras,
    }


@router.post("/import", status_code=status.HTTP_201_CREATED)
def import_camera(
    body: ImportRequest,
    user: TokenPayload = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    """
    Import a discovered ONVIF camera into the database.
    Credentials are stored separately via CredentialStore after this call.
    """
    camera = Camera(
        tenant_id=user.tenant_id,
        name=body.name,
        host=body.host,
        path=body.path,
        enabled=True,
    )
    db.add(camera)
    db.flush()

    return {
        "id": str(camera.id),
        "status": "imported",
        "note": "Use POST /api/cameras/{id}/credentials to store RTSP credentials.",
    }
