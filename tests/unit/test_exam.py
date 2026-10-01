"""Unit tests for exam proctoring signals, flags, store and API — synthetic people, no video."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from analytics.store import LessonStore
from api.auth import TokenPayload, get_current_user
from api.routes import lessons
from classroom.consent import AuditLog, ConsentRegistry
from classroom.exam import ExamConfig, ExamMonitor, room_noise_flags
from classroom.seatmap import Seat, SeatMap
from classroom.states import PersonObs

W, H = 1000, 500
FPS = 3


def keypoints(cx, nose_dx=0.0, wrist_dx=0.0):
    """Seated student, shoulders 40 px apart at y=250, facing the camera."""
    k = np.zeros((17, 3), dtype=np.float32)
    k[0] = [cx + nose_dx, 220, 0.9]
    k[1] = [cx + 8, 212, 0.9]
    k[2] = [cx - 8, 212, 0.9]
    k[3] = [cx + 16, 216, 0.8]
    k[4] = [cx - 16, 216, 0.8]
    k[5] = [cx + 20, 250, 0.9]
    k[6] = [cx - 20, 250, 0.9]
    k[9] = [cx + 15 + wrist_dx, 300, 0.9]
    k[10] = [cx - 15, 300, 0.9]
    return k


def row_seatmap(n=4, spacing=100):
    return SeatMap([Seat(f"R1C{i + 1}", (100 + i * spacing) / W, 250 / H, spacing / W / 2) for i in range(n)])


def person(i, nose_dx=0.0, wrist_dx=0.0, height=200, at_seat=True, phone=False, student=True):
    cx = 100 + i * 100
    return PersonObs(i + 1, [cx - 30, 300 - height, cx + 30, 300], keypoints(cx, nose_dx, wrist_dx),
                     f"R1C{i + 1}" if student else None, student, True, phone, at_seat)


def run(monitor, frames):
    """frames: list of per-frame people lists, at FPS."""
    for n, people in enumerate(frames):
        monitor.observe(n / FPS, people, W, H)
    return monitor.finish()


def calm(n_frames, extra=lambda f: {}):
    """4 students facing forward; `extra(frame)` overrides person kwargs by index."""
    out = []
    for f in range(n_frames):
        over = extra(f)
        out.append([person(i, **over.get(i, {})) for i in range(4)])
    return out


class TestExamMonitor:
    def test_calm_exam_raises_nothing(self):
        assert run(ExamMonitor(row_seatmap()), calm(60)) == []

    def test_sustained_gaze_flags_and_blip_does_not(self):
        # Student 1 turns to the right (towards seat R1C3) for 4 s after a 6 s baseline
        flags = run(ExamMonitor(row_seatmap()), calm(36, lambda f: {1: {"nose_dx": 16}} if 18 <= f < 30 else {}))
        assert [(f.type, f.seat_id, f.neighbour_seat) for f in flags] == [("gaze_neighbour", "R1C2", "R1C3")]
        blip = run(ExamMonitor(row_seatmap()), calm(36, lambda f: {1: {"nose_dx": 16}} if 18 <= f < 21 else {}))
        assert blip == []

    def test_angled_student_is_their_own_baseline(self):
        # Student 0 sits angled the whole exam: that's their normal, not an event
        assert run(ExamMonitor(row_seatmap()), calm(60, lambda f: {0: {"nose_dx": 16}})) == []

    def test_mutual_facing_is_talking(self):
        flags = run(ExamMonitor(row_seatmap()), calm(36, lambda f: (
            {1: {"nose_dx": 16}, 2: {"nose_dx": -16}} if f >= 18 else {})))
        assert {(f.type, f.seat_id) for f in flags} == {("talking", "R1C2"), ("talking", "R1C3")}

    def test_reaching_relative_to_own_wrist_baseline(self):
        flags = run(ExamMonitor(row_seatmap()), calm(36, lambda f: {1: {"wrist_dx": 60}} if f >= 18 else {}))
        assert [(f.type, f.neighbour_seat) for f in flags] == [("reaching", "R1C3")]

    def test_invigilator_nearby_suppresses(self):
        def frames():
            out = []
            for f in range(36):
                people = [person(i, nose_dx=16 if (i == 1 and f >= 18) else 0) for i in range(4)]
                inv = PersonObs(99, [170, 50, 230, 300], keypoints(200), None, False, False)
                out.append(people + [inv])
            return out
        assert run(ExamMonitor(row_seatmap()), frames()) == []

    def test_small_or_off_seat_students_not_assessed(self):
        small = run(ExamMonitor(row_seatmap()), calm(36, lambda f: (
            {1: {"nose_dx": 16, "height": 60}} if f >= 18 else {1: {"height": 60}})))
        assert small == []
        m = ExamMonitor(row_seatmap())
        run(m, calm(10, lambda f: {1: {"at_seat": False}}))
        assert m.summary([])["assessable_share"] == 0.75

    def test_device_flag_toggle(self):
        frames = calm(15, lambda f: {2: {"phone": True}})
        assert [f.type for f in run(ExamMonitor(row_seatmap()), frames)] == ["device"]
        assert run(ExamMonitor(row_seatmap(), ExamConfig(flag_devices=False)), frames) == []

    def test_summary_rates(self):
        m = ExamMonitor(row_seatmap())
        flags = run(m, calm(36, lambda f: {1: {"nose_dx": 16}} if 18 <= f < 30 else {}))
        s = m.summary(flags)
        assert s["flags"] == 1 and s["by_type"]["gaze_neighbour"] == 1
        assert s["student_hours"] == pytest.approx(4 * 35 / FPS / 3600, abs=1e-3)


def test_room_noise_flags():
    curve = pd.DataFrame({"noisy": [False] * 5 + [True] * 4 + [False] * 3 + [True] * 2})
    assert room_noise_flags(curve, min_seconds=3) == [{"type": "room_talking", "start_s": 5.0, "end_s": 9.0}]


# ----------------------------------------------------------------------
# Store + API
# ----------------------------------------------------------------------

@pytest.fixture
def exam_lesson(tmp_path):
    root = tmp_path / "outputs"
    d = root / "exam1"
    (d / "flags").mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"seats": 2}))
    (d / "exam_flags.json").write_text(json.dumps({"flags": [
        {"id": "F0001", "type": "reaching", "seat_id": "R1C1", "track_id": 1, "start_s": 1, "end_s": 4,
         "score": 0.9, "neighbour_seat": "R1C2", "evidence": {"duration_s": 3}, "student_id": "S1",
         "clip": "flags/F0001.mp4"},
        {"id": "F0002", "type": "gaze_neighbour", "seat_id": "R1C2", "track_id": 2, "start_s": 5, "end_s": 9,
         "score": 0.4, "neighbour_seat": None, "evidence": {"duration_s": 4}, "student_id": "S2", "clip": None},
    ], "room": []}))
    (d / "exam_summary.json").write_text(json.dumps({"flags": 2}))
    (d / "flags" / "F0001.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    reg = tmp_path / "registry.csv"
    reg.write_text("student_id,name,section,status,purposes,verified_by,verified_on\n"
                   "S1,Asha,10A,granted,exam,form,2026-09-01\nS2,Ravi,10A,granted,engagement,form,2026-09-01\n")
    return root, ConsentRegistry(reg)


def client(role, root, registry, audit):
    app = FastAPI()
    app.include_router(lessons.router)
    app.dependency_overrides[get_current_user] = lambda: TokenPayload(sub="u1", tenant_id="t", role=role, exp=0)
    app.dependency_overrides[lessons.get_store] = lambda: LessonStore(root, registry)
    app.dependency_overrides[lessons.get_audit] = lambda: audit
    return TestClient(app)


class TestExamApi:
    def test_roles_consent_and_reviews(self, exam_lesson, tmp_path):
        root, registry = exam_lesson
        audit = AuditLog(tmp_path / "audit.jsonl")
        assert client("viewer", root, registry, audit).get("/api/lessons/exam1/exam").status_code == 403

        inv = client("invigilator", root, registry, audit)
        data = inv.get("/api/lessons/exam1/exam").json()
        assert all(f["student_id"] is None for f in data["flags"])          # invigilators don't see names

        teacher = client("teacher", root, registry, audit)
        flags = {f["id"]: f for f in teacher.get("/api/lessons/exam1/exam").json()["flags"]}
        assert flags["F0001"]["student_id"] == "S1"                          # exam consent
        assert flags["F0002"]["student_id"] is None                          # engagement-only consent

        assert inv.get("/api/lessons/exam1/exam/clips/F0001").status_code == 200
        assert inv.get("/api/lessons/exam1/exam/clips/F0002").status_code == 404
        assert inv.post("/api/lessons/exam1/exam/reviews", json={"flag_id": "F0001", "decision": "dismissed",
                                                                 "note": "writing"}).status_code == 200
        assert inv.post("/api/lessons/exam1/exam/reviews", json={"flag_id": "F0001", "decision": "guilty"}).status_code == 422
        assert inv.post("/api/lessons/exam1/exam/reviews", json={"flag_id": "F9", "decision": "dismissed"}).status_code == 404
        review = inv.get("/api/lessons/exam1/exam").json()["flags"][0]["review"]
        assert review["decision"] == "dismissed" and review["reviewer"] == "u1"
        assert "review_exam_flag" in [e["action"] for e in audit.entries()]
