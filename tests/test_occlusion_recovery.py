"""Bridging a gap should follow the arc, not fly off on a tangent.

yolo_detector carried a gravity-aware predictor whose docstring called it
"what makes the gap-fill follow cross-court smashes instead of predicting
straight lines". Nothing ever called it, and gap-filling happens in the
tracker, for every detector — so a shuttle that vanished mid-flight was coasted
along a straight line while the real one fell away from it.
"""

import math
import unittest

from tracker import TrajectoryTracker


def ballistic(frames, vx=14.0, vy=-22.0, gravity=1.6, start=(150.0, 500.0)):
    """A shuttle in flight, in image coordinates where y grows downward."""
    x, y = start
    path = []
    for _ in range(frames):
        path.append((x, y))
        x += vx
        y += vy
        vy += gravity
    return path


def final_gap_error(tracker, path, visible, gap):
    for point in path[:visible]:
        tracker.add_point(point)
    error = None
    for index in range(visible, visible + gap):
        tracker.add_point(None)
        if tracker.trajectory:
            px, py = tracker.trajectory[-1]
            tx, ty = path[index]
            error = math.hypot(px - tx, py - ty)
    return error


class OcclusionRecoveryTests(unittest.TestCase):
    def test_a_bridged_gap_follows_the_arc(self) -> None:
        path = ballistic(19)
        tracker = TrajectoryTracker()
        tracker.observed_fps = 30.0
        error = final_gap_error(tracker, path, visible=10, gap=9)

        self.assertIsNotNone(error, "no prediction was made across the gap")
        # A straight line from the same point lands about 60 px out after nine
        # frames; following the arc keeps it inside half that.
        self.assertLess(error, 30.0)

    def test_the_estimate_recovers_the_acceleration(self) -> None:
        tracker = TrajectoryTracker()
        tracker.observed_fps = 30.0
        for point in ballistic(12, gravity=1.6):
            tracker.add_point(point)
        self.assertAlmostEqual(tracker.gravity_px, 1.6, delta=0.5)

    def test_the_arc_beats_a_straight_line_and_by_more_over_longer_gaps(self) -> None:
        """The whole justification, stated as a test.

        Note this is a clean parabola. A real shuttlecock has enormous drag,
        so its horizontal speed decays too and neither model is exact; the
        gain on real footage should be smaller than it is here.
        """
        def error(gravity_aware: bool, gap: int) -> float:
            path = ballistic(10 + gap)
            tracker = TrajectoryTracker()
            tracker.observed_fps = 30.0
            for point in path[:10]:
                tracker.add_point(point)
            result = 0.0
            for index in range(10, 10 + gap):
                if not gravity_aware:
                    tracker.gravity_px = 0.0
                tracker.add_point(None)
                if not gravity_aware:
                    tracker.gravity_px = 0.0
                px, py = tracker.trajectory[-1]
                tx, ty = path[index]
                result = math.hypot(px - tx, py - ty)
            return result

        short_gain = error(False, 3) - error(True, 3)
        long_gain = error(False, 9) - error(True, 9)
        self.assertGreater(short_gain, 0, "no better than a straight line")
        self.assertGreater(long_gain, short_gain,
                           "the advantage should grow with the gap")


if __name__ == "__main__":
    unittest.main()
