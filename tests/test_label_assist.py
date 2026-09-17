"""Assisted labelling must save effort without inventing observations.

The tool suggests boxes from TrackNet and can interpolate between sparse
labels. Both are ways of *not* looking at a frame, so the risk is an
evaluation set quietly full of detector output — which would make measuring
that same detector circular. These tests pin the boundaries that stop it.
"""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from label_assist import (
    DEFAULT_MAX_FILL_GAP,
    SIDECAR,
    anchor_indices,
    frame_index,
    interpolate,
    is_contiguous,
    load_sidecar,
    looks_sequential,
    save_sidecar,
)


def names(count, start=0, step=1):
    return [f"/frames/frame_{start + i * step:04d}.jpg" for i in range(count)]


class FrameNaming(unittest.TestCase):
    def test_index_comes_from_the_trailing_number(self):
        self.assertEqual(frame_index("/a/frame_0042.jpg"), 42)
        self.assertEqual(frame_index("clip_02_0007.png"), 7)

    def test_unnumbered_name_has_no_index(self):
        self.assertIsNone(frame_index("/a/photo.jpg"))


class SequentialNaming(unittest.TestCase):
    def test_consecutive_numbers_look_sequential(self):
        self.assertTrue(looks_sequential(names(10)))

    def test_gaps_do_not(self):
        self.assertFalse(looks_sequential(names(10, step=25)))

    def test_unnumbered_frames_do_not(self):
        self.assertFalse(looks_sequential(["/a/one.jpg", "/a/two.jpg"]))


def write_frames(directory, count, drift):
    """Frames whose content changes by `drift` grey levels per step."""
    paths = []
    base = np.full((64, 64, 3), 40, dtype=np.uint8)
    for i in range(count):
        frame = np.clip(base.astype(int) + i * drift, 0, 255).astype(np.uint8)
        path = os.path.join(directory, f"frame_{i:04d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


class Contiguity(unittest.TestCase):
    """Filenames cannot answer this — extract_frames.py numbers sampled
    frames 0000, 0001, 0002 even when they are minutes apart, so the test has
    to look at the pixels."""

    def test_barely_changing_frames_are_contiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_frames(directory, 12, drift=1)
            self.assertTrue(is_contiguous(paths))

    def test_sequentially_named_but_jumpy_frames_are_not(self):
        # The datasets/images case: numbered 0000-0999, sampled minutes apart.
        with tempfile.TemporaryDirectory() as directory:
            paths = write_frames(directory, 12, drift=40)
            self.assertFalse(is_contiguous(paths))

    def test_unnumbered_frames_are_not(self):
        self.assertFalse(is_contiguous(["/a/one.jpg", "/a/two.jpg"]))

    def test_unreadable_frames_are_not(self):
        # Never guess "contiguous" from filenames alone when the pixels are
        # unavailable: that is the failure this test replaced.
        self.assertFalse(is_contiguous(names(10)))

    def test_a_single_frame_is_not(self):
        self.assertFalse(is_contiguous(names(1)))


class Anchors(unittest.TestCase):
    def test_stride_one_asks_for_every_frame(self):
        self.assertEqual(anchor_indices(names(6), 1), set(range(6)))

    def test_stride_selects_every_nth(self):
        self.assertIn(3, anchor_indices(names(10), 3))
        self.assertNotIn(4, anchor_indices(names(10), 3))

    def test_last_frame_is_always_an_anchor(self):
        # Anything past the final anchor cannot be interpolated — there is no
        # label on the far side of it.
        self.assertIn(9, anchor_indices(names(10), 4))


class Interpolation(unittest.TestCase):
    def test_fills_a_short_gap_on_the_line(self):
        filled = interpolate({0: (0.0, 0.0), 3: (30.0, 60.0)})
        self.assertEqual(sorted(filled), [1, 2])
        self.assertAlmostEqual(filled[1][0], 10.0)
        self.assertAlmostEqual(filled[2][1], 40.0)

    def test_refuses_a_gap_wider_than_the_limit(self):
        # Two labels far apart say nothing reliable about what happened
        # between them.
        wide = DEFAULT_MAX_FILL_GAP + 5
        self.assertEqual(interpolate({0: (0.0, 0.0), wide: (10.0, 10.0)}), {})

    def test_never_overwrites_an_existing_label(self):
        points = {0: (0.0, 0.0), 1: (99.0, 99.0), 3: (30.0, 30.0)}
        self.assertNotIn(1, interpolate(points))

    def test_does_not_interpolate_towards_a_confirmed_empty_frame(self):
        # None means a human looked and saw no shuttle. Inventing a position
        # between two sightings that straddle it would contradict them.
        points = {0: (0.0, 0.0), 2: None, 4: (40.0, 40.0)}
        self.assertEqual(interpolate(points), {})

    def test_an_empty_neighbour_is_not_a_position(self):
        self.assertEqual(interpolate({0: None, 3: None}), {})

    def test_nothing_to_fill_between_adjacent_labels(self):
        self.assertEqual(interpolate({0: (0.0, 0.0), 1: (1.0, 1.0)}), {})


class Provenance(unittest.TestCase):
    def test_missing_sidecar_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_sidecar(directory),
                             {"accepted": [], "filled": []})

    def test_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            save_sidecar(directory, {"accepted": ["a.jpg"],
                                     "filled": ["b.jpg"]})
            self.assertEqual(load_sidecar(directory)["filled"], ["b.jpg"])

    def test_sidecar_is_json_next_to_the_labels(self):
        # An eval set has to be filterable back down to observed frames, so
        # this file has to be readable by something other than this tool.
        with tempfile.TemporaryDirectory() as directory:
            save_sidecar(directory, {"accepted": [], "filled": ["c.jpg"]})
            with open(os.path.join(directory, SIDECAR)) as handle:
                self.assertEqual(json.load(handle)["filled"], ["c.jpg"])


if __name__ == "__main__":
    unittest.main()
