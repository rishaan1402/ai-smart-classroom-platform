"""
LessonStore — read-only view over pipeline outputs for the dashboard API.

A lesson is a directory under the outputs root holding any of:
  summary.json, class_engagement.csv, seat_timeline.parquet, seatmap.json  (classroom.pipeline)
  audio/teacher_metrics.json, audio/transcript.json                      (audio.pipeline)
  topics/topics.json, topics/questions_llm.json                          (topics.segment)
  lesson_topics.csv, lesson_methods.csv                                   (analytics.lesson)

Framework-free so it can be unit-tested; api/routes/lessons.py adds auth,
role gating and audit logging on top. Student identities are only returned
when the caller passes reveal_students=True (the route decides by role) and
consent still allows the purpose.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from classroom.consent import ConsentRegistry

LESSON_ID = re.compile(r"^[A-Za-z0-9_\-]+$")
MAX_CURVE_POINTS = 600
STATE_ORDER = ["attentive", "writing", "hand_raised", "head_down", "turned_away",
               "interacting", "phone", "out_of_seat", "unknown"]
ENGAGED = {"attentive", "writing", "hand_raised"}


class LessonNotFound(Exception):
    pass


class LessonStore:
    def __init__(self, root: Path, registry: Optional[ConsentRegistry] = None) -> None:
        self.root = root
        self.registry = registry

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _dir(self, lesson_id: str) -> Path:
        if not LESSON_ID.match(lesson_id):
            raise LessonNotFound(lesson_id)
        path = (self.root / lesson_id).resolve()
        if path.parent != self.root.resolve() or not path.is_dir() or not self._is_lesson(path):
            raise LessonNotFound(lesson_id)
        return path

    @staticmethod
    def _is_lesson(path: Path) -> bool:
        return (path / "summary.json").exists() or (path / "audio" / "teacher_metrics.json").exists()

    def preview_path(self, lesson_id: str) -> Optional[Path]:
        p = self._dir(lesson_id) / "preview.mp4"
        return p if p.exists() else None

    # ------------------------------------------------------------------
    # Lessons
    # ------------------------------------------------------------------

    def list_lessons(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for path in sorted(p for p in self.root.iterdir() if p.is_dir() and self._is_lesson(p)):
            summary = _json(path / "summary.json") or {}
            audio = _json(path / "audio" / "teacher_metrics.json") or {}
            topics = _json(path / "topics" / "topics.json") or []
            out.append({
                "id": path.name,
                "has_video": bool(summary),
                "has_preview": (path / "preview.mp4").exists(),
                "has_audio": bool(audio),
                "has_topics": bool(topics),
                "duration_s": audio.get("duration_s") or summary.get("duration_analysed_s"),
                "mean_engaged_share": summary.get("mean_engaged_share"),
                "seats": summary.get("seats"),
                "talk_ratio": audio.get("talk_ratio"),
                "language": audio.get("language"),
                "topics": len(topics),
                "has_exam": (path / "exam_flags.json").exists(),
            })
        return out

    def lesson(self, lesson_id: str) -> dict:
        path = self._dir(lesson_id)
        summary = _json(path / "summary.json")
        engagement = _csv(path / "class_engagement.csv")
        transcript = _json(path / "audio" / "transcript.json")
        return {
            "id": lesson_id,
            "has_preview": (path / "preview.mp4").exists(),
            "has_exam": (path / "exam_flags.json").exists(),
            "video": _public_summary(summary),
            "engagement": _curve(engagement),
            "state_mix": _state_mix(summary),
            "teacher": _json(path / "audio" / "teacher_metrics.json"),
            "minutes": _records(_csv(path / "audio" / "minute_timeline.csv")),
            "topics": _json(path / "topics" / "topics.json") or [],
            "questions": _json(path / "topics" / "questions_llm.json") or [],
            "topic_table": _records(_csv(path / "lesson_topics.csv")),
            "methods": _records(_csv(path / "lesson_methods.csv")),
            "transcript": [
                {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
                for s in (transcript or {}).get("segments", [])
            ],
        }

    def seats(self, lesson_id: str, reveal_students: bool = False, purpose: str = "engagement") -> list[dict]:
        """Seat positions with engagement; student_id only when revealed and consent still allows it."""
        path = self._dir(lesson_id)
        seatmap = _json(path / "seatmap.json") or {"seats": []}
        timeline = _parquet(path / "seat_timeline.parquet")
        identity = _json(path / "seat_identity.json") or {}

        stats: dict[str, dict] = {}
        if timeline is not None and len(timeline):
            for seat_id, g in timeline.groupby("seat_id"):
                counts = g["state"].value_counts()
                stats[seat_id] = {
                    "seconds": int(len(g)),
                    "engaged_share": round(float(g["state"].isin(ENGAGED).mean()), 3),
                    "dominant_state": counts.index[0],
                    "state_mix": {k: round(v / len(g), 3) for k, v in counts.items()},
                }

        out = []
        for seat in seatmap["seats"]:
            sid = (identity.get(seat["seat_id"]) or {}).get("student_id")
            visible = reveal_students and sid and (self.registry is None or self.registry.allows(sid, purpose))
            out.append({
                "seat_id": seat["seat_id"], "x": seat["x"], "y": seat["y"], "radius": seat.get("radius"),
                "student_id": sid if visible else None,
                **stats.get(seat["seat_id"], {"seconds": 0, "engaged_share": None,
                                              "dominant_state": None, "state_mix": {}}),
            })
        return out

    # ------------------------------------------------------------------
    # Exam review
    # ------------------------------------------------------------------

    REVIEW_DECISIONS = ("confirmed", "dismissed", "unclear")

    def exam(self, lesson_id: str, reveal_students: bool = False) -> Optional[dict]:
        """Flags, summary and latest review per flag; student ids only when revealed and exam consent allows."""
        path = self._dir(lesson_id)
        flags_doc = _json(path / "exam_flags.json")
        if flags_doc is None:
            return None
        reviews = self.exam_reviews(lesson_id)
        flags = []
        for f in flags_doc["flags"]:
            sid = f.get("student_id")
            visible = reveal_students and sid and (self.registry is None or self.registry.allows(sid, "exam"))
            flags.append({**f, "student_id": sid if visible else None, "review": reviews.get(f["id"])})
        return {"summary": _json(path / "exam_summary.json") or {}, "flags": flags, "room": flags_doc.get("room", [])}

    def exam_clip_path(self, lesson_id: str, flag_id: str) -> Optional[Path]:
        path = self._dir(lesson_id)
        flags_doc = _json(path / "exam_flags.json") or {"flags": []}
        flag = next((f for f in flags_doc["flags"] if f["id"] == flag_id), None)
        if flag is None or not flag.get("clip"):
            return None
        clip = (path / flag["clip"]).resolve()
        return clip if clip.parent == (path / "flags").resolve() and clip.exists() else None

    def exam_reviews(self, lesson_id: str) -> dict[str, dict]:
        """Latest review per flag."""
        path = self._dir(lesson_id) / "exam_reviews.jsonl"
        latest: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    latest[r["flag_id"]] = r
        return latest

    def add_exam_review(self, lesson_id: str, flag_id: str, decision: str, note: str, reviewer: str) -> dict:
        if decision not in self.REVIEW_DECISIONS:
            raise ValueError(f"decision must be one of {self.REVIEW_DECISIONS}")
        path = self._dir(lesson_id)
        flags_doc = _json(path / "exam_flags.json") or {"flags": []}
        if flag_id not in {f["id"] for f in flags_doc["flags"]}:
            raise KeyError(flag_id)
        entry = {"flag_id": flag_id, "decision": decision, "note": note[:500], "reviewer": reviewer, "ts": time.time()}
        with (path / "exam_reviews.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    # ------------------------------------------------------------------
    # Students
    # ------------------------------------------------------------------

    def student(self, student_id: str, purpose: str = "engagement") -> Optional[dict]:
        """Per-lesson engagement for one student, or None if consent doesn't allow it."""
        if self.registry is None or not self.registry.allows(student_id, purpose):
            return None
        lessons = []
        for lesson in self.list_lessons():
            timeline = _parquet(self.root / lesson["id"] / "seat_timeline.parquet")
            if timeline is None or "student_id" not in timeline:
                continue
            mine = timeline[timeline["student_id"] == student_id]
            if not len(mine):
                continue
            counts = mine["state"].value_counts()
            lessons.append({
                "lesson_id": lesson["id"],
                "seat_id": mine["seat_id"].mode().iloc[0],
                "seconds": int(len(mine)),
                "engaged_share": round(float(mine["state"].isin(ENGAGED).mean()), 3),
                "state_mix": {k: round(v / len(mine), 3) for k, v in counts.items()},
            })
        record = self.registry.get(student_id)
        return {"student_id": student_id, "name": record.name, "section": record.section, "lessons": lessons}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _parquet(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_parquet(path) if path.exists() else None


def _records(df: Optional[pd.DataFrame]) -> list[dict]:
    if df is None:
        return []
    return json.loads(df.to_json(orient="records"))


def _public_summary(summary: Optional[dict]) -> Optional[dict]:
    """Drop local file paths before sending to the browser."""
    if not summary:
        return None
    return {k: v for k, v in summary.items() if k not in {"video"}}


def _state_mix(summary: Optional[dict]) -> list[dict]:
    dist = (summary or {}).get("state_distribution", {})
    return [{"state": s, "share": dist[s]} for s in STATE_ORDER if dist.get(s)]


def _curve(df: Optional[pd.DataFrame]) -> list[dict]:
    """Per-second engagement, downsampled to ≤ MAX_CURVE_POINTS by averaging buckets."""
    if df is None or not len(df):
        return []
    df = df.sort_values("second")
    bucket = max(1, -(-len(df) // MAX_CURVE_POINTS))
    grouped = df.groupby(df["second"] // bucket * bucket).agg(
        engaged=("engaged_share", "mean"), students=("students", "mean"),
    ).reset_index()
    return [
        {"t": int(r.second), "engaged": round(float(r.engaged), 3), "students": round(float(r.students), 1)}
        for r in grouped.itertuples()
    ]
