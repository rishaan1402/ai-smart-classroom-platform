"""
Labelling API — serves face-pixelated task crops and records ground-truth labels.

The model's prediction never leaves the server (labellers must not be anchored
by it). Labels are appended per labeller (user id) to data/labels/<lesson>/labels.jsonl.
"""

import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from labeling.tasks import LABELS, add_label, load_labels, load_tasks, next_task

from ..auth import TokenPayload, get_current_user

router = APIRouter(prefix="/api/labeling", tags=["labeling"])

LABEL_ROLES = {"admin", "principal", "teacher", "labeler"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


def labels_root() -> Path:
    return Path(os.getenv("LABELS_ROOT", "data/labels"))


def require_labeler(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    if user.role not in LABEL_ROLES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires a labelling role")
    return user


def _tasks_or_404(lesson: str, root: Path) -> dict:
    tasks = load_tasks(lesson, root) if SAFE_ID.match(lesson) else None
    if tasks is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No labelling tasks for this lesson")
    return tasks


class LabelIn(BaseModel):
    task_id: str
    label: str


@router.get("")
def list_task_sets(user: TokenPayload = Depends(require_labeler), root: Path = Depends(labels_root)):
    if not root.exists():
        return {"labels": LABELS, "lessons": []}
    out = []
    for path in sorted(p for p in root.iterdir() if p.is_dir() and (p / "tasks.json").exists()):
        tasks = load_tasks(path.name, root)
        labels = load_labels(path.name, root)
        mine = labels[labels["labeller"] == user.sub]
        out.append({"lesson": path.name, "total": len(tasks["tasks"]), "mine": int(mine["task_id"].nunique()),
                    "labellers": int(labels["labeller"].nunique())})
    return {"labels": LABELS, "lessons": out}


@router.get("/{lesson}/next")
def get_next(lesson: str, user: TokenPayload = Depends(require_labeler), root: Path = Depends(labels_root)):
    _tasks_or_404(lesson, root)
    return next_task(lesson, user.sub, root)


@router.get("/{lesson}/crops/{task_id}")
def get_crop(lesson: str, task_id: str, user: TokenPayload = Depends(require_labeler),
             root: Path = Depends(labels_root)):
    tasks = _tasks_or_404(lesson, root)
    if task_id not in {t["id"] for t in tasks["tasks"]}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown task")
    path = root / lesson / "crops" / f"{task_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crop missing")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/{lesson}/labels")
def post_label(lesson: str, body: LabelIn, user: TokenPayload = Depends(require_labeler),
               root: Path = Depends(labels_root)):
    _tasks_or_404(lesson, root)
    try:
        add_label(lesson, body.task_id, body.label, user.sub, root)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown task")
    return next_task(lesson, user.sub, root)
