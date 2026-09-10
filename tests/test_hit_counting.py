"""Hit and rally counting — the numbers on the HUD people actually judge.

Each test here corresponds to something that was wrong: a "rally" counter that
counted hits, a threshold that meant different things at different frame
rates, and predicted points feeding the reversal test that produced them.
"""

import unittest

from tracker import TrajectoryTracker


def fly(tracker, start, velocity, frames):
    """Move the shuttle in a straight line, one detection per frame."""
    x, y = start
    hits = 0
    for _ in range(frames):
        x += velocity[0]
        y += velocity[1]
        hits += bool(tracker.add_point((x, y)))
    return hits


class RallyCountingTests(unittest.TestCase):
    def test_a_rally_is_not_just_a_hit_counter(self) -> None:
        """rally_count and hit_count were incremented on the same line.

        The HUD showed them as separate fields, so "rally" read as a count of
        rallies when it was really hits since the last keypress.
        """
        tracker = TrajectoryTracker()
        tracker.observed_fps = 30.0

        fly(tracker, (100, 300), (12, -8), 8)
        fly(tracker, (196, 236), (12, 20), 8)   # reversal: one hit
        self.assertGreaterEqual(tracker.hit_count, 1)

        # No rally has finished yet — the shuttle is still in play.
        self.assertEqual(tracker.rallies_completed, 0)

    def test_losing_the_shuttle_long_enough_ends_the_rally(self) -> None:
        tracker = TrajectoryTracker()
        tracker.observed_fps = 30.0

        fly(tracker, (100, 300), (12, -8), 8)
        fly(tracker, (196, 236), (12, 20), 8)
        hits_in_first = tracker.rally_count
        self.assertGreaterEqual(hits_in_first, 1)

        # Shuttle gone: a brief gap is an occlusion, a long one is the point
        # being over.
        for _ in range(40):
            tracker.add_point(None)

        self.assertEqual(tracker.rallies_completed, 1)
        self.assertEqual(tracker.rally_lengths, [hits_in_first])

        # The next sighting starts a new rally with its own count.
        tracker.add_point((400, 400))
        self.assertEqual(tracker.rally_count, 0)
        self.assertEqual(tracker.rallies_completed, 1)

    def test_a_brief_occlusion_does_not_end_the_rally(self) -> None:
        tracker = TrajectoryTracker()
        tracker.observed_fps = 30.0
        fly(tracker, (100, 300), (12, -8), 8)

        for _ in range(3):          # well inside max_no_detection_seconds
            tracker.add_point(None)
        tracker.add_point((220, 220))

        self.assertEqual(tracker.rallies_completed, 0)


class FrameRateTests(unittest.TestCase):
    @staticmethod
    def _stroke(fps: float, step: float) -> int:
        """One vertical reversal, sampled at the given rate."""
        tracker = TrajectoryTracker()
        tracker.observed_fps = fps
        x, y = 200.0, 400.0
        for _ in range(6):
            x += step
            y -= step
            tracker.add_point((x, y))
        for _ in range(6):
            x += step
            y += step
            tracker.add_point((x, y))
        return tracker.hit_count

    def test_a_soft_stroke_is_below_threshold_at_every_frame_rate(self) -> None:
        """The discriminating case.

        Velocities are pixels per frame, so the same physical stroke reads 8
        px/frame at 30 fps and 20 at 12. Against a fixed threshold of 25 that
        stroke was under at 30 fps and over at 12 — counted or not depending
        on which detector happened to be running.
        """
        self.assertEqual(self._stroke(30.0, 8.0), self._stroke(12.0, 20.0))

    def test_a_hard_stroke_is_counted_at_every_frame_rate(self) -> None:
        """The other half: scaling must not stop real hits registering."""
        at_30 = self._stroke(30.0, 16.0)
        self.assertGreaterEqual(at_30, 1)
        self.assertEqual(at_30, self._stroke(12.0, 40.0))


class PredictedPointTests(unittest.TestCase):
    def test_a_hit_during_an_occlusion_is_still_counted(self) -> None:
        """A predicted point is the motion model extrapolating itself.

        When the shuttle is struck while briefly occluded, the bridged points
        carry on in the old direction. Comparing a real segment against those
        understates the reversal, and the hit goes uncounted. Excluding them
        compares the real points either side of the gap, which is where the
        reversal actually shows.

        This is a narrow effect — one case in a 245-point sweep of stroke
        speeds and gap lengths — but it is the case where a genuine hit is
        lost, so it is the one worth holding onto.
        """
        tracker = TrajectoryTracker()
        tracker.observed_fps = 30.0
        x, y = 200.0, 400.0
        # The stroke reverses vertical velocity by 34 px/frame, comfortably
        # over the 25 px/frame threshold. It used to reverse by 18 and still
        # register, because the old fixed smoothing overshot the turn and
        # reported a velocity change of 27 that the shuttle never made. With
        # the trajectory tracked accurately the inflation is gone, so the
        # stroke has to clear the threshold on its own to be testing what this
        # case is about: that a predicted point does not hide the reversal.
        for _ in range(6):
            x += 8
            y -= 16
            tracker.add_point((x, y))
        tracker.add_point(None)     # struck while out of sight
        before = tracker.hit_count
        for _ in range(6):
            x += 10
            y += 18
            tracker.add_point((x, y))

        self.assertGreater(tracker.hit_count, before,
                           "the hit either side of the gap was not counted")

    def test_a_gap_alone_is_not_a_hit(self) -> None:
        tracker = TrajectoryTracker()
        tracker.observed_fps = 30.0
        fly(tracker, (100, 300), (12, -8), 8)

        before = tracker.hit_count
        for _ in range(3):
            tracker.add_point(None)

        self.assertEqual(tracker.hit_count, before)
        self.assertTrue(any(tracker.point_is_predicted),
                        "the gap should have been bridged by prediction")
        # Real and predicted points stay in step, or the filter reads the
        # wrong flags.
        self.assertEqual(len(tracker.trajectory), len(tracker.point_is_predicted))


if __name__ == "__main__":
    unittest.main()


class PauseTests(unittest.TestCase):
    def test_pausing_closes_the_current_rally(self) -> None:
        """SPACE means "this is not play any more".

        Without closing it, resuming after a replay or an advert appends the
        next rally's hits to the previous one's count.
        """
        tracker = TrajectoryTracker()
        tracker.observed_fps = 30.0
        fly(tracker, (100, 300), (12, -8), 8)
        fly(tracker, (196, 236), (12, 20), 8)
        self.assertGreaterEqual(tracker.rally_count, 1)

        tracker.end_rally()
        self.assertEqual(tracker.rallies_completed, 1)

        # Resuming starts a fresh rally rather than extending the old one.
        tracker.add_point((500, 500))
        self.assertEqual(tracker.rally_count, 0)

    def test_ending_a_rally_twice_counts_it_once(self) -> None:
        tracker = TrajectoryTracker()
        tracker.observed_fps = 30.0
        fly(tracker, (100, 300), (12, -8), 8)
        fly(tracker, (196, 236), (12, 20), 8)
        tracker.end_rally()
        tracker.end_rally()
        self.assertEqual(tracker.rallies_completed, 1)
