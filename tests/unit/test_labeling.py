"""Unit tests for labelling tasks, label store, evaluation and the labelling API."""

import json

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import TokenPayload, get_current_user
from api.routes import labeling as labeling_routes
from labeling.evaluate import cohen_kappa, report
from labeling.tasks import add_label, load_labels, next_task, sample_tasks


def make_tasks(root, lesson="l1", predicted=("attentive", "writing", "head_down", "attentive"), heights=None):
    d = root / lesson
    (d / "crops").mkdir(parents=True)
    heights = heights or [60, 90, 120, 150][: len(predicted)]
    tasks = [{"id": f"t{i}", "lesson": lesson, "seat_id": f"R1C{i}", "track_id": i, "t": float(i),
              "bbox": [0, 0, 10, 10], "box_height_px": h, "predicted": p}
             for i, (p, h) in enumerate(zip(predicted, heights))]
    (d / "tasks.json").write_text(json.dumps({"lesson": lesson, "tasks": tasks, "labels": []}))
    for t in tasks:
        (d / "crops" / f"{t['id']}.jpg").write_bytes(b"\xff\xd8\xff")
    return tasks


class TestSampling:
    def test_one_per_seat_window_and_cap(self):
        tokens = pd.DataFrame({
            "seat_id": ["A"] * 20 + ["B"] * 20, "timestamp_s": list(range(20)) * 2,
            "state": ["attentive"] * 40,
        })
        out = sample_tasks(tokens, n=100, min_gap_s=5)
        assert len(out) == 8                                      # 2 seats × 4 windows
        assert out.groupby("seat_id")["timestamp_s"].apply(lambda s: (s // 5).nunique()).tolist() == [4, 4]
        assert len(sample_tasks(tokens, n=3, min_gap_s=5)) == 3


class TestLabelStore:
    def test_add_validate_and_latest_wins(self, tmp_path):
        make_tasks(tmp_path)
        add_label("l1", "t0", "writing", "alice", tmp_path)
        add_label("l1", "t0", "attentive", "alice", tmp_path)     # correction
        with pytest.raises(ValueError):
            add_label("l1", "t0", "sleepy", "alice", tmp_path)
        with pytest.raises(KeyError):
            add_label("l1", "nope", "attentive", "alice", tmp_path)
        labels = load_labels("l1", tmp_path)
        assert labels.set_index("task_id").loc["t0", "label"] == "attentive"

    def test_next_task_hides_prediction_and_tracks_progress(self, tmp_path):
        make_tasks(tmp_path)
        t = next_task("l1", "alice", tmp_path)
        assert t["id"] == "t0" and "predicted" not in t and t["done"] == 0
        add_label("l1", "t0", "attentive", "alice", tmp_path)
        assert next_task("l1", "alice", tmp_path)["id"] == "t1"
        assert next_task("l1", "bob", tmp_path)["id"] == "t0"       # per-labeller progress
        for i in range(1, 4):
            add_label("l1", f"t{i}", "attentive", "alice", tmp_path)
        assert next_task("l1", "alice", tmp_path)["id"] is None


class TestEvaluate:
    def test_metrics_and_exclusions(self, tmp_path):
        make_tasks(tmp_path, predicted=("attentive", "writing", "head_down", "attentive", "attentive", "writing"),
                   heights=[50, 60, 100, 110, 150, 160])
        for tid, lab in [("t0", "attentive"), ("t1", "writing"), ("t2", "writing"),
                         ("t3", "turned_away"), ("t4", "cant_tell"), ("t5", "not_a_student")]:
            add_label("l1", tid, lab, "alice", tmp_path)
        text, payload = report(["l1"], tmp_path)
        m = payload["metrics"]
        assert m["n"] == 4                                   # cant_tell / not_a_student excluded
        assert m["accuracy"] == 0.5                          # t0, t1 right; t2, t3 wrong
        assert m["engaged_accuracy"] == 0.5                  # t2: head_down vs writing; t3: attentive vs turned
        assert "not_a_student): 1 of 6" in text
        att = next(r for r in payload["per_class"] if r["state"] == "attentive")
        assert att["precision"] == 0.5 and att["recall"] == 1.0

    def test_majority_and_kappa(self, tmp_path):
        make_tasks(tmp_path)
        for tid, a, b in [("t0", "attentive", "attentive"), ("t1", "writing", "writing"),
                          ("t2", "head_down", "writing"), ("t3", "attentive", "attentive")]:
            add_label("l1", tid, a, "alice", tmp_path)
            add_label("l1", tid, b, "bob", tmp_path)
        _, payload = report(["l1"], tmp_path)
        assert payload["agreement"]["tasks"] == 4 and payload["agreement"]["raw_agreement"] == 0.75

    def test_kappa_values(self):
        assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
        assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


class TestLabelingApi:
    def client(self, role, root):
        app = FastAPI()
        app.include_router(labeling_routes.router)
        app.dependency_overrides[get_current_user] = lambda: TokenPayload(sub="u1", tenant_id="t", role=role, exp=0)
        app.dependency_overrides[labeling_routes.labels_root] = lambda: root
        return TestClient(app)

    def test_flow_and_gating(self, tmp_path):
        make_tasks(tmp_path)
        c = self.client("teacher", tmp_path)
        sets = c.get("/api/labeling").json()
        assert sets["lessons"][0]["total"] == 4 and "attentive" in sets["labels"]
        nxt = c.get("/api/labeling/l1/next").json()
        assert nxt["id"] == "t0" and "predicted" not in nxt
        assert c.get("/api/labeling/l1/crops/t0").status_code == 200
        assert c.get("/api/labeling/l1/crops/..%2Ftasks.json").status_code == 404
        r = c.post("/api/labeling/l1/labels", json={"task_id": "t0", "label": "writing"})
        assert r.status_code == 200 and r.json()["id"] == "t1"
        assert c.post("/api/labeling/l1/labels", json={"task_id": "t1", "label": "bogus"}).status_code == 422
        assert c.get("/api/labeling/../next").status_code == 404
        assert self.client("viewer", tmp_path).get("/api/labeling").status_code == 403
