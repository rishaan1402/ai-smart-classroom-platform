"""Unit tests for VLM flag verification — fake VLM client, tiny synthetic video."""

import cv2
import numpy as np
import pytest

from classroom.exam import ExamFlag
from classroom.pipeline import FrameDetections
from classroom.seatmap import Seat, SeatMap
from classroom.vlm_verify import QUESTION, flag_strip, verdict_for, verify_flags


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(50):
        w.write(np.full((240, 320, 3), 40 + i, dtype=np.uint8))
    w.release()
    return path


def frames():
    return [FrameDetections(i * 3, i * 0.3, 320, 240, [7], [[100, 80, 140, 180]], [np.zeros((17, 3))])
            for i in range(17)]


def seatmap():
    return SeatMap([Seat("R1C1", 0.375, 0.5, 0.05), Seat("R1C2", 0.6, 0.5, 0.05)])


def flag(fid="F0001", kind="gaze_neighbour", score=0.8):
    return ExamFlag(fid, kind, "R1C1", 7, 1.0, 3.0, score, neighbour_seat="R1C2")


class FakeVLM:
    model = "fake-vl"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat_json(self, system, user, schema, temperature=0.1, images=None):
        self.calls.append((user, images))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_strip_has_three_panels(video):
    strip = flag_strip(video, flag(), frames(), seatmap())
    assert strip is not None and strip.shape[0] == 448 and strip.shape[1] == 3 * (448 + 8)


def test_verdicts_set_priority_and_neutral_question(video):
    flags = [flag("F1", score=0.8), flag("F2", "reaching", 0.8), flag("F3", "device", 0.8)]
    client = FakeVLM([
        {"description": "head down, writing on own sheet", "activity": "writing_on_own_paper"},
        {"description": "hand passes a slip to the right", "activity": "passing_or_receiving_object"},
        {"description": "blurry", "activity": "something_else"},       # outside the list → unclear
    ])
    stats = verify_flags(video, flags, frames(), seatmap(), client)
    assert [f.vlm["verdict"] for f in flags] == ["normal", "suspicious", "unclear"]
    assert flags[0].priority == pytest.approx(0.2) and flags[1].priority == pytest.approx(0.8)
    assert all(call[0] == QUESTION for call in client.calls)          # same question whatever the flag type
    assert "neighbour" not in QUESTION and len(client.calls[1][1]) == 1
    assert stats["normal"] == 1 and stats["suspicious"] == 1 and stats["unclear"] == 1


def test_description_check_overrides_inconsistent_label(video):
    flags = [flag("F1"), flag("F2")]
    vlm = FakeVLM([
        {"description": "looking down at their own table", "activity": "looking_towards_neighbour"},
        {"description": "turned to the right, looking at the neighbour's sheet", "activity": "looking_towards_neighbour"},
    ])
    check = FakeVLM([{"activity": "writing_on_own_paper"}, {"activity": "looking_towards_neighbour"}])
    check.model = "fake-text"
    stats = verify_flags(video, flags, frames(), seatmap(), vlm, check_client=check)
    assert flags[0].vlm["verdict"] == "normal" and flags[0].vlm["disagree"] and flags[0].vlm["vlm_activity"] == "looking_towards_neighbour"
    assert flags[1].vlm["verdict"] == "suspicious" and not flags[1].vlm["disagree"]
    assert stats["label_description_disagree"] == 1 and stats["normal"] == 1 and stats["suspicious"] == 1
    assert check.calls[0][0] == "looking down at their own table"


def test_check_only_reexamines_suspicious_labels(video):
    flags = [flag("F1"), flag("F2")]
    vlm = FakeVLM([
        {"description": "holding a pen, writing", "activity": "writing_on_own_paper"},
        {"description": "looking at own hands", "activity": "looking_towards_neighbour"},
    ])
    check = FakeVLM([{"activity": "handling_own_items"}])
    check.model = "fake-text"
    verify_flags(video, flags, frames(), seatmap(), vlm, check_client=check)
    assert len(check.calls) == 1 and check.calls[0][0] == "looking at own hands"   # normal label left alone
    assert [f.vlm["verdict"] for f in flags] == ["normal", "normal"]

    recheck = FakeVLM([{"activity": "looking_towards_neighbour"}])
    recheck.model = "fake-text"
    verify_flags(video, flags, frames(), seatmap(), FakeVLM([]), check_client=recheck, reuse_vlm=True)
    assert [f.vlm["verdict"] for f in flags] == ["normal", "suspicious"]         # no VLM calls needed


def test_verdict_mapping():
    assert verdict_for("using_calculator") == "normal"
    assert verdict_for("using_phone") == "suspicious"
    assert verdict_for("cannot_tell") == "unclear"


def test_errors_and_limit_keep_rule_score(video):
    flags = [flag("F1", score=0.7), flag("F2", score=0.6)]
    stats = verify_flags(video, flags, frames(), seatmap(), FakeVLM([TimeoutError("down")]), limit=1)
    assert stats["errors"] == 1
    assert all(f.vlm is None for f in flags)
    assert [f.priority for f in flags] == [0.7, 0.6]
