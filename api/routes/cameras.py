"""
Camera management API — CRUD + live test endpoint.
"""

import os
from typing import Optional
from uuid import UUID

import cv2
import redis as redis_lib
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import TokenPayload, get_current_user, require_role
from ..deps import get_db
from ..models import Camera

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


class CameraCreate(BaseModel):
    name: str
    host: str
    path: str
    enabled: bool = True
    zone_config: Optional[dict] = None


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    host: Optional[str] = None
    path: Optional[str] = None
    enabled: Optional[bool] = None
    zone_config: Optional[dict] = None


@router.get("")
def list_cameras(
    user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.execute(select(Camera).order_by(Camera.created_at)).scalars().all()
    return [
        {
            "id": str(c.id),
            "name": c.name,
            "host": c.host,
            "path": c.path,
            "enabled": c.enabled,
            "zone_config": c.zone_config,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in rows
    ]


@router.post("", status_code=status.HTTP_201_CREATED)
def create_camera(
    body: CameraCreate,
    user: TokenPayload = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    camera = Camera(
        tenant_id=user.tenant_id,
        name=body.name,
        host=body.host,
        path=body.path,
        enabled=body.enabled,
        zone_config=body.zone_config,
    )
    db.add(camera)
    db.flush()

    # Notify engine of camera config change
    _notify_camera_change()

    return {"id": str(camera.id), "status": "created"}


@router.put("/{camera_id}")
def update_camera(
    camera_id: UUID,
    body: CameraUpdate,
    user: TokenPayload = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    cam = db.execute(select(Camera).where(Camera.id == camera_id)).scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    if body.name is not None:
        cam.name = body.name
    if body.host is not None:
        cam.host = body.host
    if body.path is not None:
        cam.path = body.path
    if body.enabled is not None:
        cam.enabled = body.enabled
    if body.zone_config is not None:
        cam.zone_config = body.zone_config

    db.flush()
    _notify_camera_change()
    return {"id": str(cam.id), "status": "updated"}


@router.delete("/{camera_id}")
def delete_camera(
    camera_id: UUID,
    user: TokenPayload = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    cam = db.execute(select(Camera).where(Camera.id == camera_id)).scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    db.delete(cam)
    db.flush()
    _notify_camera_change()
    return {"status": "deleted"}


@router.post("/{camera_id}/test")
def test_camera(
    camera_id: UUID,
    user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Grab a single frame from the camera RTSP stream.
    Returns JPEG thumbnail + status within 5s timeout.
    """
    cam = db.execute(select(Camera).where(Camera.id == camera_id)).scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    rtsp_url = f"rtsp://{cam.host}{cam.path}"
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000",
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    try:
        if not cap.isOpened():
            return {"status": "error", "detail": "Cannot connect to camera"}

        ret, frame = cap.read()
        if not ret or frame is None:
            return {"status": "error", "detail": "No frame received"}

        # Encode as JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return StreamingResponse(
            iter([buf.tobytes()]),
            media_type="image/jpeg",
            headers={"X-Camera-Status": "online"},
        )
    finally:
        cap.release()


def _notify_camera_change() -> None:
    """Publish camera config change to Redis for engine hot-reload."""
    try:
        r = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
        )
        r.publish("camera_config_changed", "1")
        r.close()
    except Exception:
        pass
