"""
Events API — list, filter by severity, single event detail.
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import TokenPayload, get_current_user
from ..deps import get_db
from ..models import Event

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("")
def list_events(
    severity: Optional[str] = Query(None, description="critical,high,medium,low (comma-separated)"),
    camera_id: Optional[UUID] = None,
    event_type: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = Query(50, ge=1, le=500),
    user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List events for the current tenant (RLS scoped).

    Default filter: CRITICAL + HIGH severity in the last 24h.
    """
    q = select(Event).order_by(Event.created_at.desc()).limit(limit)

    if severity:
        levels = [s.strip().lower() for s in severity.split(",")]
        q = q.where(Event.severity.in_(levels))
    else:
        q = q.where(Event.severity.in_(["critical", "high"]))

    if camera_id:
        q = q.where(Event.camera_id == camera_id)
    if event_type:
        q = q.where(Event.event_type == event_type)
    if since:
        q = q.where(Event.created_at >= since)
    else:
        q = q.where(Event.created_at >= datetime.utcnow() - timedelta(days=1))

    rows = db.execute(q).scalars().all()
    return [
        {
            "id": str(e.id),
            "trace_id": str(e.trace_id),
            "schema_version": e.schema_version,
            "event_type": e.event_type,
            "severity": e.severity,
            "person_id": e.person_id,
            "camera_id": str(e.camera_id) if e.camera_id else None,
            "face_id": str(e.face_id) if e.face_id else None,
            "face_recognition_was_active": e.face_recognition_was_active,
            "face_display": _face_display(e),
            "enrichment_skipped": e.enrichment_skipped,
            "threat_level": e.threat_level,
            "snapshot_path": e.snapshot_path,
            "clip_path": e.clip_path,
            "metadata": e.metadata_,
            "created_at": e.created_at.isoformat(),
        }
        for e in rows
    ]


@router.get("/{event_id}")
def get_event(
    event_id: UUID,
    user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    e = db.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
    if not e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return {
        "id": str(e.id),
        "trace_id": str(e.trace_id),
        "event_type": e.event_type,
        "severity": e.severity,
        "threat_level": e.threat_level,
        "face_display": _face_display(e),
        "snapshot_path": e.snapshot_path,
        "clip_path": e.clip_path,
        "metadata": e.metadata_,
        "created_at": e.created_at.isoformat(),
    }


def _face_display(event: Event) -> str:
    """
    Resolve face display name from face_id + face_recognition_was_active.

    Matrix:
      face_id set                        → look up name (handled at UI)
      face_id NULL + was_active=true     → "[anonymized]"
      face_id NULL + was_active=false    → "face detection off"
    """
    if event.face_id:
        return "identified"
    if event.face_recognition_was_active:
        return "[anonymized]"
    return "face detection off"
