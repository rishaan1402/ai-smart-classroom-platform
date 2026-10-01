"""Unit tests for crop-level pose refinement helpers — no model, no video."""

import numpy as np
import pytest

from classroom.refine import crop_box, merge_keypoints, needs_refinement, pick_target


def kps(arm_conf=0.9):
    k = np.full((17, 3), 0.9, dtype=np.float32)
    k[[7, 8, 9, 10], 2] = arm_conf
    return k


class TestGates:
    def test_arms_gate(self):
        assert needs_refinement(kps(0.2), [0, 0, 50, 100], "arms")
        assert not needs_refinement(kps(0.9), [0, 0, 50, 100], "arms")

    def test_small_and_all(self):
        assert needs_refinement(kps(), [0, 0, 50, 100], "small")
        assert not needs_refinement(kps(), [0, 0, 50, 200], "small")
        assert needs_refinement(kps(), [0, 0, 50, 400], "all")
        with pytest.raises(ValueError):
            needs_refinement(kps(), [0, 0, 1, 1], "bogus")


class TestCropAndPick:
    def test_crop_box_clamps_to_frame(self):
        assert crop_box([5, 5, 105, 205], 100, 200) == (0, 0, 100, 200)
        x1, y1, x2, y2 = crop_box([100, 100, 200, 300], 1000, 1000)
        assert (x1, y1, x2, y2) == (70, 50, 230, 350)

    def test_pick_target_prefers_overlap_and_rejects_strangers(self):
        boxes = np.array([[0, 0, 40, 40], [50, 50, 150, 150]], dtype=np.float32)
        assert pick_target(boxes, (55, 55, 145, 145)) == 1
        assert pick_target(boxes, (300, 300, 400, 400)) is None
        assert pick_target(np.empty((0, 4)), (0, 0, 1, 1)) is None


class TestMerge:
    def test_more_confident_joint_wins(self):
        original, refined = kps(0.2), kps(0.7)
        refined[:, :2] = 5.0
        refined[0, 2] = 0.1                       # refined nose worse → keep original nose
        merged = merge_keypoints(original, refined)
        assert merged[9, 2] == pytest.approx(0.7) and merged[9, 0] == 5.0
        assert merged[0, 2] == pytest.approx(0.9) and merged[0, 0] == original[0, 0]
