"""Smoothing must not be spent where it costs accuracy.

The tracker used to blend a fixed fraction of the one-step prediction into
every detection. Measured against the labelled clips that cost accuracy at
every setting -- 9.0 px median error at the shipped default against 4.0 px for
the raw detections -- because a linear-plus-gravity prediction cannot keep up
with a shuttlecock and blending towards it just adds lag.

These tests pin the shape of the replacement: lean on the prediction only when
it is both mature and in agreement with what was detected.
"""

import unittest

from tracker import TrajectoryTracker


def tracker(smoothing=0.8, adaptive=True):
    t = TrajectoryTracker(max_trajectory_length=60, hit_detection_threshold=25,
                          smoothing=smoothing, adaptive_smoothing=adaptive)
    t.set_frame_size(1280, 720)
    return t


def fly(t, start, velocity, frames):
    """Move the shuttle in a straight line, one detection per frame."""
    x, y = start
    for _ in range(frames):
        x += velocity[0]
        y += velocity[1]
        t.add_point((x, y))
    return x, y


class Maturity(unittest.TestCase):
    def test_first_detection_is_never_smoothed(self):
        # There is nothing to predict from yet.
        t = tracker()
        t.add_point((100.0, 100.0))
        self.assertEqual(t.trajectory[-1], (100, 100))

    def test_a_young_track_leans_on_the_detection(self):
        # Two frames in, the velocity estimate is built from a single step.
        t = tracker()
        t.add_point((100.0, 100.0))
        t.add_point((110.0, 100.0))
        self.assertLess(t._effective_smoothing((200.0, 200.0), (110.0, 100.0)),
                        0.3)

    def test_maturity_rises_with_tracked_frames(self):
        t = tracker()
        fly(t, (100.0, 100.0), (4.0, 0.0), 2)
        young = t.frames_since_acquired
        fly(t, (108.0, 100.0), (4.0, 0.0), 6)
        self.assertGreater(t.frames_since_acquired, young)

    def test_reacquisition_after_leaving_frame_resets_maturity(self):
        # The velocity estimate starts again, so the prediction is worthless
        # again -- and a stale one would smear the trail across the court.
        t = tracker()
        fly(t, (600.0, 300.0), (10.0, 0.0), 10)
        self.assertGreater(t.frames_since_acquired, 4)
        t.add_point((1275.0, 300.0))  # inside the edge margin
        self.assertTrue(t.out_of_frame)
        t.add_point((300.0, 300.0))   # reappears elsewhere
        # Reset, then counted as this frame's own detection.
        self.assertEqual(t.frames_since_acquired, 1)
        # What that is for: the track is young again, so the prediction
        # carries almost no weight however far it misses.
        self.assertLess(t._effective_smoothing((900.0, 300.0), (310.0, 300.0)),
                        0.05)


class Agreement(unittest.TestCase):
    def test_disagreement_pulls_smoothing_down(self):
        t = tracker()
        fly(t, (100.0, 300.0), (5.0, 0.0), 12)
        close = t._effective_smoothing((200.0, 300.0), (201.0, 300.0))
        far = t._effective_smoothing((200.0, 300.0), (260.0, 300.0))
        self.assertLess(far, close)

    def test_a_far_miss_all_but_ignores_the_prediction(self):
        # A hit or a direction change: the motion model has been overtaken.
        t = tracker()
        fly(t, (100.0, 300.0), (5.0, 0.0), 12)
        self.assertLess(t._effective_smoothing((200.0, 300.0), (400.0, 300.0)),
                        0.05)

    def test_agreement_never_exceeds_the_configured_ceiling(self):
        t = tracker(smoothing=0.6)
        fly(t, (100.0, 300.0), (5.0, 0.0), 12)
        # An exact agreement is the best case, and it is still capped.
        self.assertLessEqual(
            t._effective_smoothing((200.0, 300.0), (200.0, 300.0)), 0.6 + 1e-9)

    def test_smoothing_stays_within_bounds_across_a_rally(self):
        t = tracker()
        for velocity in ((5.0, 2.0), (-30.0, 12.0), (18.0, -25.0)):
            fly(t, (400.0, 300.0), velocity, 8)
            s = t._effective_smoothing((410.0, 305.0), (415.0, 302.0))
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 0.8 + 1e-9)


class OptOut(unittest.TestCase):
    def test_disabling_restores_the_fixed_blend(self):
        t = tracker(smoothing=0.5, adaptive=False)
        fly(t, (100.0, 300.0), (5.0, 0.0), 6)
        # Whatever the residual, the fixed blend is exactly the setting.
        self.assertEqual(t._effective_smoothing((200.0, 300.0), (900.0, 20.0)),
                         0.5)
        self.assertEqual(t._effective_smoothing((200.0, 300.0), (200.0, 300.0)),
                         0.5)

    def test_adaptive_is_the_default(self):
        self.assertTrue(TrajectoryTracker().adaptive_smoothing)


class AccuracyAgainstFixed(unittest.TestCase):
    def test_adaptive_tracks_a_smash_closer_than_a_fixed_blend(self):
        # The case the measurement is about: fast motion, where a fixed blend
        # lags behind the shuttle and adaptive does not.
        start, velocity, frames = (200.0, 100.0), (40.0, 22.0), 10

        def final_error(adaptive):
            t = tracker(smoothing=0.8, adaptive=adaptive)
            x, y = fly(t, start, velocity, frames)
            px, py = t.trajectory[-1]
            return ((px - x) ** 2 + (py - y) ** 2) ** 0.5

        self.assertLess(final_error(True), final_error(False))


if __name__ == "__main__":
    unittest.main()
