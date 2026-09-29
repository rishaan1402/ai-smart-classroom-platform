"""Unit tests for consent, gallery, seat identification, blurring and purge — fake embedder, no models."""

import json

import numpy as np
import pandas as pd
import pytest
from cryptography.fernet import Fernet

from classroom.consent import AuditLog, ConsentRegistry
from classroom.faces import Face
from classroom.identity import Gallery, SeatIdentifier, blur_heads, head_box
from classroom.privacy import enroll, purge
from sentinel_engine.workers.face import FaceIndex

KEY = Fernet.generate_key()


# ----------------------------------------------------------------------
# Fakes: an "image" of student k is a uniform patch with pixel value k*10
# ----------------------------------------------------------------------

def identity_vector(k: int, dim: int = 128) -> np.ndarray:
    v = np.random.default_rng(k).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    dim = 128

    def __init__(self, noise: float = 0.05):
        self.noise = noise
        self.rng = np.random.default_rng(99)

    def detect(self, image):
        if image.mean() < 5:
            return []
        h, w = image.shape[:2]
        lm = np.array([[w * .3, h * .4], [w * .7, h * .4], [w * .5, h * .6], [w * .35, h * .8], [w * .65, h * .8]])
        return [Face((0, 0, w, h), lm, 0.95, np.zeros(15))]

    def embed(self, image, face):
        k = int(round(image.mean() / 10))
        v = identity_vector(k) + self.noise * self.rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)


def patch(k: int, size: int = 100) -> np.ndarray:
    return np.full((size, size, 3), k * 10, dtype=np.uint8)


def registry(tmp_path, rows) -> ConsentRegistry:
    path = tmp_path / "registry.csv"
    lines = ["student_id,name,section,status,purposes,verified_by,verified_on"]
    lines += [f"{sid},Name {sid},8A,{status},{purposes},signed form,2026-09-01" for sid, status, purposes in rows]
    path.write_text("\n".join(lines) + "\n")
    return ConsentRegistry(path)


# ----------------------------------------------------------------------
# Consent
# ----------------------------------------------------------------------

class TestConsent:
    def test_allows_only_granted_with_purpose(self, tmp_path):
        reg = registry(tmp_path, [("S1", "granted", "engagement;exam"), ("S2", "granted", "exam"),
                                  ("S3", "pending", "engagement"), ("S4", "withdrawn", "")])
        assert reg.allows("S1", "engagement") and reg.allows("S1", "exam")
        assert not reg.allows("S2", "engagement")
        assert not reg.allows("S3", "engagement") and not reg.allows("S4", "engagement")
        assert not reg.allows("unknown", "engagement")
        assert reg.allowed_students("engagement") == {"S1"}

    def test_withdraw_persists(self, tmp_path):
        reg = registry(tmp_path, [("S1", "granted", "engagement")])
        reg.withdraw("S1")
        reg.save()
        assert not ConsentRegistry(reg.path).allows("S1", "engagement")

    def test_bad_status_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            registry(tmp_path, [("S1", "maybe", "engagement")])

    def test_audit_log_appends(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("enroll", "S1", embeddings=3)
        log.record("purge", "S1")
        assert [e["action"] for e in log.entries()] == ["enroll", "purge"]


# ----------------------------------------------------------------------
# Gallery & FaceIndex
# ----------------------------------------------------------------------

class TestGallery:
    def test_encrypted_roundtrip_and_wrong_key(self, tmp_path):
        g = Gallery(128)
        g.add("S1", [identity_vector(1)])
        g.save(tmp_path / "g.enc", KEY)
        assert b"students" not in (tmp_path / "g.enc").read_bytes()    # no plaintext JSON on disk
        assert Gallery.load(tmp_path / "g.enc", KEY).students == {"S1"}
        with pytest.raises(Exception):
            Gallery.load(tmp_path / "g.enc", Fernet.generate_key())

    def test_search_and_restrict(self):
        g = Gallery(128)
        g.add("S1", [identity_vector(1)])
        g.add("S2", [identity_vector(2)])
        assert g.search(identity_vector(2), 0.363)[0] == "S2"
        assert g.search(identity_vector(7), 0.363) is None
        g.restrict_to({"S1"})
        assert g.students == {"S1"} and g.search(identity_vector(2), 0.363) is None

    def test_face_index_normalizes_on_add(self):
        idx = FaceIndex(embedding_dim=4)
        idx.initialize()
        idx.add("a", "A", np.array([3.0, 0, 0, 0]))          # not unit length
        hit = idx.search(np.array([1.0, 0, 0, 0]), threshold=0.9)
        assert hit is not None and hit[2] == pytest.approx(1.0, abs=1e-5)


class TestEnroll:
    def test_refused_without_consent(self, tmp_path):
        reg = registry(tmp_path, [("S1", "pending", "engagement")])
        photos = tmp_path / "photos"
        photos.mkdir()
        audit = AuditLog(tmp_path / "audit.jsonl")
        with pytest.raises(PermissionError):
            enroll(reg, tmp_path / "g.enc", audit, "S1", photos, embedder=FakeEmbedder())
        assert audit.entries()[-1]["action"] == "enroll_refused"
        assert not (tmp_path / "g.enc").exists()

    def test_enrolls_with_consent(self, tmp_path, monkeypatch):
        import cv2
        monkeypatch.setenv("FACE_ENCRYPTION_KEY", KEY.decode())
        reg = registry(tmp_path, [("S1", "granted", "engagement")])
        photos = tmp_path / "photos"
        photos.mkdir()
        for i in range(3):
            cv2.imwrite(str(photos / f"{i}.png"), patch(1))
        used = enroll(reg, tmp_path / "g.enc", AuditLog(tmp_path / "a.jsonl"), "S1", photos, embedder=FakeEmbedder())
        assert used == 3
        assert Gallery.load(tmp_path / "g.enc").students == {"S1"}


# ----------------------------------------------------------------------
# Seat identification
# ----------------------------------------------------------------------

def identifier_with(students=(1, 2)) -> SeatIdentifier:
    g = Gallery(128)
    for k in students:
        g.add(f"S{k}", [identity_vector(k)])
    return SeatIdentifier(g, FakeEmbedder(), min_face_px=10, min_votes=3)


def observe_n(ident, seat, k, n):
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[100:200, 100:200] = k * 10
    for _ in range(n):
        ident.observe(frame, seat, (100, 100, 200, 200))


class TestSeatIdentifier:
    def test_face_votes_identify_seat(self):
        ident = identifier_with()
        observe_n(ident, "R1C1", 1, 5)
        out = ident.resolve(["R1C1", "R1C2"])
        assert out["R1C1"].student_id == "S1" and out["R1C1"].source == "face"
        assert out["R1C2"].student_id is None and out["R1C2"].source == "none"

    def test_face_and_chart_agree_or_conflict(self):
        ident = identifier_with()
        observe_n(ident, "R1C1", 1, 5)
        observe_n(ident, "R1C2", 2, 5)
        out = ident.resolve(["R1C1", "R1C2"], seating_chart={"R1C1": "S1", "R1C2": "S1"})
        assert out["R1C1"].source == "face+chart" and out["R1C1"].confidence > 0.5
        assert out["R1C2"].source == "conflict" and out["R1C2"].student_id is None

    def test_chart_only_and_unmatched_face_vetoes_chart(self):
        ident = identifier_with()
        observe_n(ident, "R1C2", 9, 5)             # clear face of someone not enrolled
        out = ident.resolve(["R1C1", "R1C2"], seating_chart={"R1C1": "S1", "R1C2": "S2"})
        assert out["R1C1"].source == "chart" and out["R1C1"].student_id == "S1"
        assert out["R1C2"].source == "conflict"

    def test_withdrawn_consent_excluded_at_resolve(self):
        ident = identifier_with()
        observe_n(ident, "R1C1", 1, 5)
        out = ident.resolve(["R1C1"], seating_chart={"R1C1": "S1"}, allowed={"S2"})
        assert out["R1C1"].student_id is None

    def test_too_few_votes_stays_anonymous(self):
        ident = identifier_with()
        observe_n(ident, "R1C1", 1, 2)
        assert ident.resolve(["R1C1"])["R1C1"].student_id is None

    def test_only_votes_are_kept_never_embeddings(self):
        ident = identifier_with()
        observe_n(ident, "R1C1", 9, 3)
        for vote in ident._votes["R1C1"]:
            assert vote == (None, 0.0)

    def test_empty_gallery_does_no_face_work(self):
        class Exploding(FakeEmbedder):
            def detect(self, image):
                raise AssertionError("face detection must not run without consented enrollments")
        ident = SeatIdentifier(Gallery(128), Exploding())
        observe_n(ident, "R1C1", 1, 3)
        assert ident.resolve(["R1C1"])["R1C1"].source == "none"


# ----------------------------------------------------------------------
# Blurring, head boxes, purge
# ----------------------------------------------------------------------

class TestBlurAndHeads:
    def test_blur_changes_only_the_box(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        before = frame.copy()
        blur_heads(frame, [(50, 50, 120, 120)])
        assert not np.array_equal(frame[50:120, 50:120], before[50:120, 50:120])
        assert np.array_equal(frame[130:, 130:], before[130:, 130:])

    def test_head_box_from_keypoints_and_fallback(self):
        kps = np.zeros((17, 3))
        kps[0] = [100, 50, 0.9]
        kps[1] = [110, 45, 0.9]
        kps[2] = [90, 45, 0.9]
        x1, y1, x2, y2 = head_box(kps, [60, 30, 140, 230])
        assert x1 < 90 and x2 > 110 and y1 < 45 and y2 > 50
        assert head_box(np.zeros((17, 3)), [0, 0, 100, 200]) == (0, 0, 100, 60)


class TestOutsideAll:
    def test_identified_heads_are_kept(self):
        from classroom.identity import outside_all
        keep = [(0, 0, 50, 50)]
        assert not outside_all((10, 10, 30, 30), keep)
        assert outside_all((60, 60, 80, 80), keep)
        assert outside_all((10, 10, 30, 30), [])


class TestPurge:
    def test_purge_removes_every_trace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FACE_ENCRYPTION_KEY", KEY.decode())
        reg = registry(tmp_path, [("S1", "granted", "engagement"), ("S2", "granted", "engagement")])
        g = Gallery(128)
        g.add("S1", [identity_vector(1)])
        g.add("S2", [identity_vector(2)])
        g.save(tmp_path / "g.enc")

        lesson = tmp_path / "outputs" / "lesson1"
        lesson.mkdir(parents=True)
        pd.DataFrame({"seat_id": ["R1C1", "R1C2"], "student_id": ["S1", "S2"], "state": ["attentive"] * 2}) \
            .to_parquet(lesson / "tokens.parquet")
        (lesson / "seat_identity.json").write_text(json.dumps({
            "R1C1": {"seat_id": "R1C1", "student_id": "S1", "source": "face", "confidence": 0.8,
                     "face_votes": 5, "candidates": {"S1": 5}},
        }))

        audit = AuditLog(tmp_path / "audit.jsonl")
        purge(reg, tmp_path / "g.enc", audit, "S1", tmp_path / "outputs")

        assert Gallery.load(tmp_path / "g.enc").students == {"S2"}
        tokens = pd.read_parquet(lesson / "tokens.parquet")
        assert "S1" not in set(tokens["student_id"].dropna()) and "S2" in set(tokens["student_id"])
        seat = json.loads((lesson / "seat_identity.json").read_text())["R1C1"]
        assert seat["student_id"] is None and "S1" not in seat["candidates"]
        assert not ConsentRegistry(reg.path).allows("S1", "engagement")
        assert audit.entries()[-1]["action"] == "purge"
