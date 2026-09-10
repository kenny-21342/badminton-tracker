"""A denied screen capture must be told apart from a real desktop.

macOS reports a Screen Recording denial by handing back a correctly sized,
perfectly blank frame rather than an error. Before this check the tracker ran
against that blank frame and looked like a detector that could not find
anything -- the failure was indistinguishable from a bad model.

These cases are synthetic because the real denial cannot be reproduced once the
permission is granted on the development machine.
"""

import unittest

import numpy as np

from screen_permission import (
    check_capture_permitted,
    frame_is_blank,
    permission_message,
)


def blank(width=1920, height=1080, value=0):
    """What macOS returns when the capture is denied: one flat colour."""
    return np.full((height, width, 4), value, dtype=np.uint8)


def desktop(width=1920, height=1080, seed=0):
    """A stand-in for a real screen: many distinct pixel values."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)


class FakeGrabber:
    def __init__(self, frame):
        self.frame = frame

    def grab(self, monitor):
        return self.frame


class RaisingGrabber:
    def grab(self, monitor):
        raise RuntimeError("capture unavailable")


class BlankFrameDetection(unittest.TestCase):
    def test_all_black_frame_is_blank(self):
        self.assertTrue(frame_is_blank(blank(value=0)))

    def test_all_white_frame_is_blank(self):
        # The denial is a flat frame, not specifically a black one.
        self.assertTrue(frame_is_blank(blank(value=255)))

    def test_real_desktop_is_not_blank(self):
        self.assertFalse(frame_is_blank(desktop()))

    def test_near_uniform_desktop_is_not_blank(self):
        # A solid-colour wallpaper is still not a denial: a menu bar, cursor or
        # window edge puts several distinct colours on screen. Anything with a
        # handful of colours must be allowed through, or the check would refuse
        # to run on a tidy desktop.
        frame = blank(value=30)
        frame[0, :40] = 200
        frame[1, :10] = 90
        frame[2, :5] = 150
        self.assertFalse(frame_is_blank(frame))

    def test_sparse_content_survives_subsampling(self):
        # The count subsamples for speed. Content must not be missed just
        # because it occupies a small part of the frame.
        frame = blank(value=10)
        frame[100:400, 100:900] = desktop(800, 300)
        self.assertFalse(frame_is_blank(frame))


class PermissionCheck(unittest.TestCase):
    def test_blank_capture_is_refused_with_a_message(self):
        ok, message = check_capture_permitted(FakeGrabber(blank()), {})
        self.assertFalse(ok)
        self.assertIn("Screen Recording", message)

    def test_real_capture_is_permitted_with_no_message(self):
        ok, message = check_capture_permitted(FakeGrabber(desktop()), {})
        self.assertTrue(ok)
        self.assertIsNone(message)

    def test_a_raising_grabber_is_refused_not_propagated(self):
        # An outright exception is also a denial as far as the user is
        # concerned, and must not surface as a traceback.
        ok, message = check_capture_permitted(RaisingGrabber(), {})
        self.assertFalse(ok)
        self.assertIn("capture unavailable", message)
        self.assertIn("Screen Recording", message)

    def test_message_says_to_restart(self):
        # The permission does not apply to an already-running process, so the
        # instruction is useless without the restart step.
        self.assertIn("restart", permission_message().lower())


if __name__ == "__main__":
    unittest.main()
