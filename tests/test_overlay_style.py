"""The point overlay must be quieter than the trail without being dishonest.

On broadcast footage the full trail is a long streak across the frame, much of
it extrapolated through gaps, and it reads as clutter over the thing it points
at. Point style draws only where the shuttle is now.

The catch it has to survive: TrackNet sees the shuttle on 61% of in-play frames
in clip_01. Gap filling carries a marker to 86%, so point style keeps it — and
therefore has to distinguish a detection from a bridged guess, which the trail
never had to.
"""

import unittest

import numpy as np

from visualizer import TrajectoryVisualizer


def blank(width=320, height=240):
    return np.zeros((height, width, 3), dtype=np.uint8)


def painted(frame):
    """How many pixels the overlay touched."""
    return int(np.count_nonzero(frame.any(axis=2)))


def a_trail(length=25, start=(40, 40), step=(8, 5)):
    return [(start[0] + i * step[0], start[1] + i * step[1])
            for i in range(length)]


class PointStyleIsQuieter(unittest.TestCase):
    def test_point_style_paints_far_less_than_the_trail(self):
        trail = a_trail()
        visualiser = TrajectoryVisualizer()

        with_trail = blank()
        visualiser.draw(with_trail, trail, current_point=trail[-1])

        visualiser.style = "point"
        with_point = blank()
        visualiser.draw(with_point, trail, current_point=trail[-1])

        self.assertLess(painted(with_point), painted(with_trail))

    def test_point_style_draws_nothing_without_a_position(self):
        # The honest state during a long dropout: no marker rather than one
        # left behind at a stale position.
        visualiser = TrajectoryVisualizer()
        visualiser.style = "point"
        frame = blank()
        visualiser.draw(frame, a_trail(), current_point=None)
        self.assertEqual(painted(frame), 0)

    def test_point_style_ignores_the_trail_it_is_given(self):
        # Same marker whether or not there is history behind it.
        visualiser = TrajectoryVisualizer()
        visualiser.style = "point"
        with_history, without = blank(), blank()
        visualiser.draw(with_history, a_trail(), current_point=(150, 120))
        visualiser.draw(without, [], current_point=(150, 120))
        self.assertEqual(painted(with_history), painted(without))


class PredictedPositionsLookDifferent(unittest.TestCase):
    def test_a_bridged_position_is_drawn_hollow(self):
        # A gap-filled point is the motion model talking, not a sighting. It
        # must not look like one.
        visualiser = TrajectoryVisualizer()
        visualiser.style = "point"
        detected, guessed = blank(), blank()
        visualiser.draw(detected, [], current_point=(150, 120),
                        current_is_predicted=False)
        visualiser.draw(guessed, [], current_point=(150, 120),
                        current_is_predicted=True)
        self.assertLess(painted(guessed), painted(detected))

    def test_trail_style_marks_predictions_too(self):
        # The distinction belongs to the marker, not to one style.
        visualiser = TrajectoryVisualizer()
        trail = a_trail()
        detected, guessed = blank(), blank()
        visualiser.draw(detected, trail, current_point=trail[-1],
                        current_is_predicted=False)
        visualiser.draw(guessed, trail, current_point=trail[-1],
                        current_is_predicted=True)
        self.assertNotEqual(painted(guessed), painted(detected))

    def test_default_is_a_detection(self):
        # Callers that never pass the flag keep the old solid marker.
        visualiser = TrajectoryVisualizer()
        visualiser.style = "point"
        default, explicit = blank(), blank()
        visualiser.draw(default, [], current_point=(150, 120))
        visualiser.draw(explicit, [], current_point=(150, 120),
                        current_is_predicted=False)
        self.assertEqual(painted(default), painted(explicit))


class StyleDefault(unittest.TestCase):
    def test_trail_remains_the_default(self):
        self.assertEqual(TrajectoryVisualizer().style, "trail")


if __name__ == "__main__":
    unittest.main()
