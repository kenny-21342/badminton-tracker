"""Session recording must actually produce a file.

SessionRecorder was fully written, constructed in main_screen and handed to
the overlay — and nothing ever called start_recording, record_frame or
stop_recording. No session was ever written.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from session_recorder import SessionRecorder
from tracker import TrajectoryTracker


class SessionExportTests(unittest.TestCase):
    def test_a_recorded_session_round_trips_with_its_rallies(self) -> None:
        with TemporaryDirectory() as directory:
            recorder = SessionRecorder(sessions_dir=Path(directory))
            tracker = TrajectoryTracker()
            tracker.observed_fps = 30.0

            recorder.start_recording("classic")
            x, y = 200.0, 400.0
            for _ in range(6):
                x += 16
                y -= 16
                tracker.add_point((x, y))
                recorder.record_frame((x, y), tracker.last_velocity,
                                      tracker.current_speed(),
                                      tracker.hit_count, tracker.rally_count)
            for _ in range(6):
                x += 16
                y += 16
                tracker.add_point((x, y))
                recorder.record_frame((x, y), tracker.last_velocity,
                                      tracker.current_speed(),
                                      tracker.hit_count, tracker.rally_count)
            for _ in range(40):        # rally ends
                tracker.add_point(None)

            path = recorder.stop_recording(tracker)
            self.assertIsNotNone(path, "stop_recording wrote nothing")

            data = json.loads(Path(path).read_text())
            self.assertEqual(data["detector_mode"], "classic")
            self.assertEqual(len(data["frames"]), 12)
            self.assertGreaterEqual(data["hits"], 1)
            # The rally summary is the part a player reads, and it only exists
            # once the session is over.
            self.assertEqual(data["rallies_completed"], tracker.rallies_completed)
            self.assertEqual(data["rally_lengths"], tracker.rally_lengths)

    def test_stopping_without_starting_is_harmless(self) -> None:
        with TemporaryDirectory() as directory:
            recorder = SessionRecorder(sessions_dir=Path(directory))
            self.assertIsNone(recorder.stop_recording())
            self.assertEqual(list(Path(directory).glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
