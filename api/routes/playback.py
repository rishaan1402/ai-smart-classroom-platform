"""
Phase 5: Recording playback API.

GET /api/playback/timeline          → Events with clips in a time range (scrub bar data)
GET /api/playback/clip/{event_id}   → Stream or redirect to clip (MP4)
GET /api/playback/snapshot/{event_id} → Stream or redirect to snapshot (JPEG)

Clips are stored in MinIO (production) or local filesystem (dev).
Signed URLs are issued for MinIO so the browser streams directly — API
is never a media proxy in production.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import TokenPayload, get_current_user
from ..deps import get_db
from ..models import Event

router = APIRouter(prefix="/api/playback", tags=["playback"])

_STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")
_LOCAL_STORAGE = os.getenv("LOCAL_STORAGE_PATH", "data/storage")
_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
_MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
_MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
_MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
_SIGNED_URL_EXPIRY = 3600  # 1 hour


@router.get("/timeline")
def timeline(
    camera_id: Optional[UUID] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = Query(200, ge=1, le=1000),
    user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return events that have clip or snapshot paths, ordered by time.
    This is the scrub-bar data source for the playback UI.
    """
    if since is None:
        since = datetime.utcnow() - timedelta(hours=24)
    if until is None:
        until = datetime.utcnow()

    q = (
        select(Event)
        .where(
            (Event.clip_path.isnot(None)) | (Event.snapshot_path.isnot(None)),
            Event.created_at >= since,
            Event.created_at <= until,
        )
        .order_by(Event.created_at.asc())
        .limit(limit)
    )

    if camera_id:
        q = q.where(Event.camera_id == camera_id)

    rows = db.execute(q).scalars().all()
    return [
        {
            "id": str(e.id),
            "event_type": e.event_type,
            "severity": e.severity,
            "camera_id": str(e.camera_id) if e.camera_id else None,
            "has_clip": e.clip_path is not None,
            "has_snapshot": e.snapshot_path is not None,
            "created_at": e.created_at.isoformat(),
            "ts_ms": int(e.created_at.timestamp() * 1000),
        }
        for e in rows
    ]


@router.get("/clip/{event_id}")
def get_clip(
    event_id: UUID,
    user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Serve or redirect to the clip for an event.
    - MinIO (production): returns a signed redirect URL (browser streams directly)
    - Local (dev): streams the file through the API
    """
    e = db.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
    if not e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    if not e.clip_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No clip for this event")

    return _serve_file(e.clip_path, media_type="video/mp4")


@router.get("/snapshot/{event_id}")
def get_snapshot(
    event_id: UUID,
    user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Serve or redirect to the snapshot JPEG for an event."""
    e = db.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
    if not e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    if not e.snapshot_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No snapshot for this event")

    return _serve_file(e.snapshot_path, media_type="image/jpeg")


def _serve_file(path: str, media_type: str):
    """
    Route file serving based on storage backend.

    path format:
      local: "snapshots/{tenant_id}/{filename}"  → LOCAL_STORAGE_PATH/path
      minio: "sentinel-snapshots/{tenant_id}/{filename}"  → signed URL redirect
    """
    if _STORAGE_BACKEND == "minio":
        url = _minio_signed_url(path)
        return RedirectResponse(url=url, status_code=302)

    # Local storage
    local_path = os.path.join(_LOCAL_STORAGE, path)
    if not os.path.exists(local_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found on disk")

    return FileResponse(local_path, media_type=media_type)


def _minio_signed_url(path: str) -> str:
    """
    Generate a pre-signed MinIO URL for the given path.
    path should be: {bucket}/{object_key}
    """
    try:
        from minio import Minio
        from datetime import timedelta

        client = Minio(
            _MINIO_ENDPOINT,
            access_key=_MINIO_ACCESS_KEY,
            secret_key=_MINIO_SECRET_KEY,
            secure=_MINIO_SECURE,
        )

        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid MinIO path: {path}")

        bucket, object_key = parts
        url = client.presigned_get_object(
            bucket,
            object_key,
            expires=timedelta(seconds=_SIGNED_URL_EXPIRY),
        )
        return url
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate signed URL: {e}",
        )
