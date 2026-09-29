"""Unit tests for LessonStore and the lessons API — synthetic outputs, no DB/Redis."""

import json

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from analytics.store import LessonNotFound, LessonStore
from api.auth import TokenPayload, get_current_user
from api.routes import lessons
from classroom.consent import AuditLog, ConsentRegistry


@pytest.fixture
def outputs(tmp_path):
    root = tmp_path / "outputs"
    lesson = root / "class8a_math"
    (lesson / "audio").mkdir(parents=True)
    (lesson / "topics").mkdir()
    (lesson / "summary.json").write_text(json.dumps({
        "video": "/secret/local/path.mp4", "seats": 2, "mean_engaged_share": 0.75, "duration_analysed_s": 120,
        "state_distribution": {"attentive": 0.6, "turned_away": 0.4},
    }))
    pd.DataFrame({"second": range(1200), "students": [2] * 1200, "engaged_share": [0.5] * 1200}) \
        .to_csv(lesson / "class_engagement.csv", index=False)
    (lesson / "seatmap.json").write_text(json.dumps({"seats": [
        {"seat_id": "R1C1", "x": 0.2, "y": 0.5, "radius": 0.05},
        {"seat_id": "R1C2", "x": 0.4, "y": 0.5, "radius": 0.05},
    ]}))
    pd.DataFrame({
        "seat_id": ["R1C1"] * 4 + ["R1C2"] * 4, "second": list(range(4)) * 2,
        "state": ["attentive"] * 3 + ["turned_away"] + ["turned_away"] * 4,
        "student_id": ["S1"] * 4 + ["S2"] * 4,
    }).to_parquet(lesson / "seat_timeline.parquet")
    (lesson / "seat_identity.json").write_text(json.dumps({
        "R1C1": {"student_id": "S1"}, "R1C2": {"student_id": "S2"},
    }))
    (lesson / "audio" / "teacher_metrics.json").write_text(json.dumps({"duration_s": 120, "talk_ratio": 0.7}))
    (lesson / "audio" / "transcript.json").write_text(json.dumps({"segments": [
        {"start": 0.0, "end": 2.5, "text": "Good morning", "words": [{"word": "Good"}]},
    ]}))
    (lesson / "topics" / "topics.json").write_text(json.dumps([{"start_s": 0, "end_s": 120, "title": "Fractions"}]))
    (lesson / "preview.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (root / "not_a_lesson").mkdir()
    return root


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "registry.csv"
    path.write_text("student_id,name,section,status,purposes,verified_by,verified_on\n"
                    "S1,Asha,8A,granted,engagement,form,2026-09-01\n"
                    "S2,Ravi,8A,withdrawn,,form,2026-09-01\n")
    return ConsentRegistry(path)


class TestStore:
    def test_list_skips_non_lessons(self, outputs, registry):
        items = LessonStore(outputs, registry).list_lessons()
        assert [i["id"] for i in items] == ["class8a_math"]
        assert items[0]["has_video"] and items[0]["has_audio"] and items[0]["topics"] == 1

    def test_lesson_hides_paths_and_downsamples(self, outputs, registry):
        data = LessonStore(outputs, registry).lesson("class8a_math")
        assert "video" not in data["video"]
        assert len(data["engagement"]) <= 600
        assert "words" not in data["transcript"][0]
        assert [s["state"] for s in data["state_mix"]] == ["attentive", "turned_away"]

    def test_bad_ids_rejected(self, outputs, registry):
        store = LessonStore(outputs, registry)
        for bad in ["../outputs", "not_a_lesson", "missing", "a/b"]:
            with pytest.raises(LessonNotFound):
                store.lesson(bad)

    def test_seats_reveal_only_consented(self, outputs, registry):
        store = LessonStore(outputs, registry)
        hidden = store.seats("class8a_math", reveal_students=False)
        assert all(s["student_id"] is None for s in hidden)
        shown = {s["seat_id"]: s for s in store.seats("class8a_math", reveal_students=True)}
        assert shown["R1C1"]["student_id"] == "S1" and shown["R1C1"]["engaged_share"] == 0.75
        assert shown["R1C2"]["student_id"] is None          # S2 withdrew consent

    def test_student_requires_consent(self, outputs, registry):
        store = LessonStore(outputs, registry)
        s1 = store.student("S1")
        assert s1["name"] == "Asha" and s1["lessons"][0]["engaged_share"] == 0.75
        assert store.student("S2") is None and store.student("S9") is None


def client_for(role, outputs, registry, audit):
    app = FastAPI()
    app.include_router(lessons.router)
    app.dependency_overrides[get_current_user] = lambda: TokenPayload(sub="u1", tenant_id="t1", role=role, exp=0)
    app.dependency_overrides[lessons.get_store] = lambda: LessonStore(outputs, registry)
    app.dependency_overrides[lessons.get_audit] = lambda: audit
    return TestClient(app)


class TestRoutes:
    def test_viewer_sees_aggregates_not_students(self, outputs, registry, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")
        c = client_for("viewer", outputs, registry, audit)
        assert c.get("/api/lessons").status_code == 200
        assert c.get("/api/lessons/class8a_math").status_code == 200
        seats = c.get("/api/lessons/class8a_math/seats").json()
        assert not seats["can_see_students"] and all(s["student_id"] is None for s in seats["seats"])
        assert c.get("/api/students/S1").status_code == 403
        assert [e["action"] for e in audit.entries()] == ["view_lesson", "view_seats", "view_student_denied"]

    def test_teacher_sees_consented_students(self, outputs, registry, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")
        c = client_for("teacher", outputs, registry, audit)
        seats = c.get("/api/lessons/class8a_math/seats").json()["seats"]
        assert {s["student_id"] for s in seats} == {"S1", None}
        assert c.get("/api/students/S1").status_code == 200
        assert c.get("/api/students/S2").status_code == 404      # withdrawn looks like unknown
        assert audit.entries()[-2]["actor"] == "u1"

    def test_preview_and_missing(self, outputs, registry, tmp_path):
        c = client_for("viewer", outputs, registry, AuditLog(tmp_path / "a.jsonl"))
        r = c.get("/api/lessons/class8a_math/preview")
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
        assert c.get("/api/lessons/nope/preview").status_code == 404
        assert c.get("/api/lessons/..%2Fsecrets").status_code == 404
