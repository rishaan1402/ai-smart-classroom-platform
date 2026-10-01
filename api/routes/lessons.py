"""
Lessons & students API — classroom analytics for the dashboard.

Reads pipeline outputs through analytics.store.LessonStore (file-based for
the pilot; one school per deployment, so not yet tenant-scoped like the
Postgres-backed routes).

Access rules:
  - Any signed-in user: lesson list, lesson aggregates, anonymous seat grid,
    blurred preview video.
  - STUDENT_ROLES only: student identities on the seat grid and per-student
    pages, and only for students whose consent still allows the purpose.
  - EXAM_ROLES only: exam flags, review clips and review decisions.
  - Every lesson view, preview stream, student view, exam view and review is audit-logged.
"""

import os
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analytics.store import LessonNotFound, LessonStore
from classroom.consent import AuditLog, ConsentRegistry

from ..auth import TokenPayload, get_current_user

router = APIRouter(prefix="/api", tags=["lessons"])

STUDENT_ROLES = {"admin", "principal", "teacher"}
EXAM_ROLES = {"admin", "principal", "teacher", "invigilator"}


@lru_cache(maxsize=1)
def _audit() -> AuditLog:
    return AuditLog(Path(os.getenv("AUDIT_LOG", "data/audit/audit.jsonl")))


def get_store() -> LessonStore:
    # Registry is re-read per request so a withdrawal takes effect immediately
    registry = ConsentRegistry(Path(os.getenv("CONSENT_REGISTRY", "data/consent/registry.csv")))
    return LessonStore(Path(os.getenv("CLASSROOM_OUTPUTS", "outputs")), registry)


def get_audit() -> AuditLog:
    return _audit()


def _can_see_students(user: TokenPayload) -> bool:
    return user.role in STUDENT_ROLES


@router.get("/lessons")
def list_lessons(user: TokenPayload = Depends(get_current_user), store: LessonStore = Depends(get_store)):
    return store.list_lessons()


@router.get("/lessons/{lesson_id}")
def get_lesson(
    lesson_id: str,
    user: TokenPayload = Depends(get_current_user),
    store: LessonStore = Depends(get_store),
    audit: AuditLog = Depends(get_audit),
):
    try:
        data = store.lesson(lesson_id)
    except LessonNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found")
    audit.record("view_lesson", actor=user.sub, role=user.role, lesson=lesson_id)
    return data


@router.get("/lessons/{lesson_id}/seats")
def get_seats(
    lesson_id: str,
    user: TokenPayload = Depends(get_current_user),
    store: LessonStore = Depends(get_store),
    audit: AuditLog = Depends(get_audit),
):
    reveal = _can_see_students(user)
    try:
        seats = store.seats(lesson_id, reveal_students=reveal)
    except LessonNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found")
    audit.record("view_seats", actor=user.sub, role=user.role, lesson=lesson_id,
                 students_revealed=sum(1 for s in seats if s["student_id"]))
    return {"can_see_students": reveal, "seats": seats}


@router.get("/lessons/{lesson_id}/preview")
def get_preview(
    lesson_id: str,
    user: TokenPayload = Depends(get_current_user),
    store: LessonStore = Depends(get_store),
    audit: AuditLog = Depends(get_audit),
):
    """The blurred preview only; unblurred video is never served by this API."""
    try:
        path = store.preview_path(lesson_id)
    except LessonNotFound:
        path = None
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No preview for this lesson")
    audit.record("view_preview", actor=user.sub, role=user.role, lesson=lesson_id)
    return FileResponse(path, media_type="video/mp4")


class ExamReviewIn(BaseModel):
    flag_id: str
    decision: str
    note: str = ""


def _require_exam_role(user: TokenPayload) -> None:
    if user.role not in EXAM_ROLES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires an exam review role")


@router.get("/lessons/{lesson_id}/exam")
def get_exam(
    lesson_id: str,
    user: TokenPayload = Depends(get_current_user),
    store: LessonStore = Depends(get_store),
    audit: AuditLog = Depends(get_audit),
):
    _require_exam_role(user)
    try:
        data = store.exam(lesson_id, reveal_students=_can_see_students(user))
    except LessonNotFound:
        data = None
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No exam flags for this lesson")
    audit.record("view_exam_flags", actor=user.sub, role=user.role, lesson=lesson_id, flags=len(data["flags"]))
    return data


@router.get("/lessons/{lesson_id}/exam/clips/{flag_id}")
def get_exam_clip(
    lesson_id: str,
    flag_id: str,
    user: TokenPayload = Depends(get_current_user),
    store: LessonStore = Depends(get_store),
    audit: AuditLog = Depends(get_audit),
):
    _require_exam_role(user)
    try:
        path = store.exam_clip_path(lesson_id, flag_id)
    except LessonNotFound:
        path = None
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No clip for this flag")
    audit.record("view_exam_clip", actor=user.sub, role=user.role, lesson=lesson_id, flag=flag_id)
    return FileResponse(path, media_type="video/mp4")


@router.post("/lessons/{lesson_id}/exam/reviews")
def post_exam_review(
    lesson_id: str,
    body: ExamReviewIn,
    user: TokenPayload = Depends(get_current_user),
    store: LessonStore = Depends(get_store),
    audit: AuditLog = Depends(get_audit),
):
    _require_exam_role(user)
    try:
        entry = store.add_exam_review(lesson_id, body.flag_id, body.decision, body.note, user.sub)
    except LessonNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found")
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown flag")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    audit.record("review_exam_flag", actor=user.sub, role=user.role, lesson=lesson_id,
                 flag=body.flag_id, decision=body.decision)
    return entry


@router.get("/students/{student_id}")
def get_student(
    student_id: str,
    user: TokenPayload = Depends(get_current_user),
    store: LessonStore = Depends(get_store),
    audit: AuditLog = Depends(get_audit),
):
    if not _can_see_students(user):
        audit.record("view_student_denied", student_id, actor=user.sub, role=user.role)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires teacher, principal or admin")
    data = store.student(student_id)
    if data is None:
        # Same response for "unknown" and "no consent": don't leak which students exist
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    audit.record("view_student", student_id, actor=user.sub, role=user.role)
    return data
