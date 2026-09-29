"""Unit tests for classroom seat map, head pose and state rules — no GPU, no video."""

import numpy as np
import pandas as pd

from classroom.headpose import estimate_head_pose
from classroom.pipeline import aggregate, phone_in_hands
from classroom.seatmap import Seat, SeatMap, SeatTracker
from classroom.states import ClassifierConfig, PersonObs, StateClassifier, TemporalSmoother
from classroom.tokens import StudentState


def seated_keypoints(cx=100.0, nose_dx=0.0, nose_dy=0.0, face_conf=0.9, wrist_y=None, wrist_dx=0.0) -> np.ndarray:
    """Upper-body COCO keypoints for a seated student facing the camera, shoulders 40 px apart."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[0] = [cx + nose_dx, 60 + nose_dy, face_conf]   # nose
    kp[1] = [cx + 8, 52, face_conf]                   # left eye (image right)
    kp[2] = [cx - 8, 52, face_conf]                   # right eye
    kp[3] = [cx + 16, 56, 0.8]                        # left ear
    kp[4] = [cx - 16, 56, 0.8]                        # right ear
    kp[5] = [cx + 20, 90, 0.9]                        # left shoulder
    kp[6] = [cx - 20, 90, 0.9]                        # right shoulder
    wy = 130 if wrist_y is None else wrist_y
    kp[9] = [cx + 15 + wrist_dx, wy, 0.9]             # left wrist
    kp[10] = [cx - 15, wy, 0.9]                       # right wrist
    return kp


def bbox_for(cx=100.0) -> list:
    return [cx - 30, 30, cx + 30, 160]


def grid_seatmap(n=4, spacing=0.1) -> SeatMap:
    seats = [Seat(f"R1C{i + 1}", 0.1 + i * spacing, 0.5, spacing / 2) for i in range(n)]
    return SeatMap(seats)


def person(tid, cx, kps, seat=None, seated=True, phone=False) -> PersonObs:
    return PersonObs(tid, bbox_for(cx), kps, seat or f"R1C{tid}", True, seated, phone)


# ----------------------------------------------------------------------
# Head pose
# ----------------------------------------------------------------------

class TestHeadPose:
    def test_frontal_face_has_near_zero_yaw(self):
        h = estimate_head_pose(seated_keypoints(), bbox_for())
        assert abs(h.yaw) < 0.05
        assert h.pitch_ratio > 0.5
        assert not h.back_turned

    def test_turned_head_has_large_yaw_with_sign(self):
        right = estimate_head_pose(seated_keypoints(nose_dx=14), bbox_for())
        left = estimate_head_pose(seated_keypoints(nose_dx=-14), bbox_for())
        assert right.yaw > 0.3 and left.yaw < -0.3

    def test_head_down_lowers_pitch(self):
        up = estimate_head_pose(seated_keypoints(), bbox_for())
        down = estimate_head_pose(seated_keypoints(nose_dy=25), bbox_for())
        assert down.pitch_ratio < 0.5 * up.pitch_ratio

    def test_back_of_head_detected(self):
        kp = seated_keypoints(face_conf=0.05)
        h = estimate_head_pose(kp, bbox_for())
        assert h.back_turned and h.yaw is None

    def test_hidden_face_at_shoulder_level_is_not_back_turned(self):
        kp = seated_keypoints(face_conf=0.05)
        kp[3, 1] = kp[4, 1] = 96   # ears below the shoulders: head down on the desk
        assert not estimate_head_pose(kp, bbox_for()).back_turned


# ----------------------------------------------------------------------
# Seat map
# ----------------------------------------------------------------------

class TestSeatMap:
    def test_calibrate_finds_seats_and_drops_walker(self):
        rng = np.random.default_rng(0)
        seats = np.array([[0.2, 0.4], [0.4, 0.4], [0.2, 0.7], [0.4, 0.7]])
        frames = []
        for f in range(40):
            pts = seats + rng.normal(0, 0.004, seats.shape)
            walker = np.array([[0.1 + f * 0.02, 0.2]])  # teacher crossing the room
            frames.append(np.vstack([pts, walker]))
        sm = SeatMap.calibrate(frames, min_occupancy=0.3)
        assert len(sm) == 4
        assert {s.seat_id for s in sm.seats} == {"R1C1", "R1C2", "R2C1", "R2C2"}

    def test_assign_is_one_to_one_and_rejects_far_points(self):
        sm = grid_seatmap(n=2)
        out = sm.assign({1: (0.1, 0.5), 2: (0.105, 0.5), 3: (0.9, 0.9)})
        assert list(out.values()).count("R1C1") == 1   # two claimants, one seat
        assert out[3] is None                           # far from every seat

    def test_json_roundtrip(self, tmp_path):
        sm = grid_seatmap()
        sm.to_json(tmp_path / "s.json")
        loaded = SeatMap.from_json(tmp_path / "s.json")
        assert [s.seat_id for s in loaded.seats] == [s.seat_id for s in sm.seats]

    def test_seat_tracker_roles(self):
        st = SeatTracker(window=5, student_share=0.5)
        for _ in range(10):
            st.update({1: "R1C1", 2: None})
        assert st.is_student(1) and not st.is_student(2)
        assert st.currently_seated(1) and not st.currently_seated(2)


# ----------------------------------------------------------------------
# State rules
# ----------------------------------------------------------------------

def class_frame(target_kps, target_phone=False, seated=True):
    """Target student (track 1) plus three attentive classmates facing forward."""
    xs = [100, 200, 300, 400]
    people = [person(1, xs[0], target_kps, seated=seated, phone=target_phone)]
    people += [person(i + 1, xs[i], seated_keypoints(cx=xs[i])) for i in range(1, 4)]
    return people


def classify(target_kps, warmup=8, **kw) -> StudentState:
    clf = StateClassifier(grid_seatmap(), ClassifierConfig())
    # Warm-up with an upright pose to build the pitch baseline
    for _ in range(warmup):
        clf.classify_frame(class_frame(seated_keypoints()))
    return clf.classify_frame(class_frame(target_kps, **kw))[1].state


class TestStateRules:
    def test_attentive(self):
        assert classify(seated_keypoints()) == StudentState.ATTENTIVE

    def test_turned_away_relative_to_class(self):
        assert classify(seated_keypoints(nose_dx=16)) == StudentState.TURNED_AWAY

    def test_hand_raised(self):
        assert classify(seated_keypoints(wrist_y=30)) == StudentState.HAND_RAISED

    def test_phone(self):
        assert classify(seated_keypoints(), target_phone=True) == StudentState.PHONE

    def test_out_of_seat(self):
        assert classify(seated_keypoints(), seated=False) == StudentState.OUT_OF_SEAT

    def test_head_resting_vs_writing(self):
        clf = StateClassifier(grid_seatmap(), ClassifierConfig())
        for _ in range(8):
            clf.classify_frame(class_frame(seated_keypoints()))
        idle = [clf.classify_frame(class_frame(seated_keypoints(nose_dy=36)))[1].state for _ in range(4)]
        assert idle[-1] == StudentState.HEAD_DOWN

        clf = StateClassifier(grid_seatmap(), ClassifierConfig())
        for _ in range(8):
            clf.classify_frame(class_frame(seated_keypoints()))
        states = [
            clf.classify_frame(class_frame(seated_keypoints(nose_dy=18, wrist_dx=(-1) ** i * 6)))[1].state
            for i in range(4)
        ]
        assert states[-1] == StudentState.WRITING

    def test_interacting_with_neighbour(self):
        clf = StateClassifier(grid_seatmap(), ClassifierConfig())
        people = [
            person(1, 100, seated_keypoints(cx=100, nose_dx=16)),    # faces image right → seat 2
            person(2, 200, seated_keypoints(cx=200, nose_dx=-16)),   # faces image left → seat 1
            person(3, 300, seated_keypoints(cx=300)),
            person(4, 400, seated_keypoints(cx=400)),
            person(5, 500, seated_keypoints(cx=500), seat="R1C4"),
        ]
        out = clf.classify_frame(people)
        assert out[1].state == StudentState.INTERACTING
        assert out[2].state == StudentState.INTERACTING

    def test_consensus_is_local_for_wide_angle_views(self):
        # Left half of the room faces the board at yaw +0.3, right half at −0.3 (perspective)
        seats = [Seat(f"R1C{i + 1}", 0.1 * (i + 1), 0.5, 0.05) for i in range(8)]
        clf = StateClassifier(SeatMap(seats), ClassifierConfig())
        people = [
            person(i + 1, 100 * (i + 1), seated_keypoints(cx=100 * (i + 1), nose_dx=12 if i < 4 else -12))
            for i in range(8)
        ]
        out = clf.classify_frame(people)
        assert all(c.state == StudentState.ATTENTIVE for c in out.values())

    def test_non_students_are_skipped(self):
        clf = StateClassifier(grid_seatmap(), ClassifierConfig())
        teacher = PersonObs(9, bbox_for(), seated_keypoints(), None, False, False)
        assert 9 not in clf.classify_frame([teacher])


class TestSmoothingAndAggregation:
    def test_smoother_majority(self):
        sm = TemporalSmoother(window=5)
        for s in [StudentState.ATTENTIVE] * 4 + [StudentState.TURNED_AWAY]:
            state, share = sm.update(1, s)
        assert state == StudentState.ATTENTIVE and share == 0.8

    def test_phone_in_hands(self):
        kps = seated_keypoints()
        assert phone_in_hands(bbox_for(), kps, [[95, 120, 110, 140]])
        assert not phone_in_hands(bbox_for(), kps, [[500, 500, 520, 520]])

    def test_aggregate_engagement(self):
        df = pd.DataFrame({
            "timestamp_s": [0.1, 0.4, 0.2, 0.5],
            "track_id": [1, 1, 2, 2],
            "seat_id": ["R1C1", "R1C1", "R1C2", "R1C2"],
            "state": ["attentive", "attentive", "turned_away", "turned_away"],
        })
        timeline, engagement = aggregate(df)
        assert len(timeline) == 2
        assert engagement.loc[0, "engaged_share"] == 0.5
