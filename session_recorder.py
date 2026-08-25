import json
import time
from pathlib import Path
from datetime import datetime


class SessionRecorder:
    """
    Records trajectory data to JSON files for later replay.
    Sessions are saved to ~/.badminton-tracker/sessions/
    """

    def __init__(self, sessions_dir=None):
        if sessions_dir is None:
            self.sessions_dir = Path.home() / ".badminton-tracker" / "sessions"
        else:
            self.sessions_dir = Path(sessions_dir)

        # Create sessions directory if it doesn't exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Recording state
        self.is_recording = False
        self.session_data = None
        self.start_time = None
        self.frame_count = 0

    def start_recording(self, detector_mode):
        """Start a new recording session."""
        if self.is_recording:
            return

        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0

        self.session_data = {
            "timestamp": datetime.now().isoformat(),
            "detector_mode": detector_mode,
            "duration_seconds": 0,
            "hits": 0,
            "rally_count": 0,
            # Filled in at stop from the tracker: one entry per completed
            # rally. This is the part a player actually reads — how long the
            # rallies were — and it needs the whole session to be known.
            "rallies_completed": 0,
            "rally_lengths": [],
            "frames": []
        }

    def stop_recording(self, tracker=None):
        """Stop recording and save to disk.

        ``tracker`` supplies the rally summary, which only exists once the
        session is over.
        """
        if not self.is_recording:
            return None

        self.is_recording = False
        if tracker is not None:
            self.session_data["rallies_completed"] = getattr(
                tracker, "rallies_completed", 0)
            self.session_data["rally_lengths"] = list(
                getattr(tracker, "rally_lengths", []))

        # Calculate duration
        self.session_data["duration_seconds"] = time.time() - self.start_time

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
        filepath = self.sessions_dir / filename

        # Save to disk
        try:
            with open(filepath, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            return filepath
        except IOError as e:
            print(f"Error saving session: {e}")
            return None

    def record_frame(self, trajectory_point, velocity, speed, hit_count, rally_count):
        """Record a single frame of data."""
        if not self.is_recording:
            return

        frame_time = time.time() - self.start_time

        frame_data = {
            "t": round(frame_time, 3),
            "x": trajectory_point[0] if trajectory_point else None,
            "y": trajectory_point[1] if trajectory_point else None,
            "vx": velocity[0] if velocity else None,
            "vy": velocity[1] if velocity else None,
            "speed": round(speed, 2) if speed else None
        }

        self.session_data["frames"].append(frame_data)
        self.session_data["hits"] = hit_count
        self.session_data["rally_count"] = rally_count
        self.frame_count += 1

    def list_sessions(self):
        """List all saved session files."""
        sessions = []
        for filepath in sorted(self.sessions_dir.glob("session_*.json"), reverse=True):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        "filename": filepath.name,
                        "path": str(filepath),
                        "timestamp": data.get("timestamp", "Unknown"),
                        "duration": data.get("duration_seconds", 0),
                        "hits": data.get("hits", 0),
                        "rally_count": data.get("rally_count", 0),
                        "detector_mode": data.get("detector_mode", "unknown")
                    })
            except (json.JSONDecodeError, IOError):
                continue

        return sessions

    def load_session(self, filepath):
        """Load a session from disk."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading session: {e}")
            return None

    def export_to_csv(self, session_data, output_path):
        """Export session data to CSV format."""
        try:
            import csv

            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'x', 'y', 'vx', 'vy', 'speed'])

                for frame in session_data["frames"]:
                    writer.writerow([
                        frame.get('t', ''),
                        frame.get('x', ''),
                        frame.get('y', ''),
                        frame.get('vx', ''),
                        frame.get('vy', ''),
                        frame.get('speed', '')
                    ])

            return True
        except IOError as e:
            print(f"Error exporting to CSV: {e}")
            return False


class SessionPlayer:
    """
    Replays recorded sessions with adjustable playback speed.
    """

    def __init__(self, session_data):
        self.session_data = session_data
        self.frames = session_data.get("frames", [])
        self.current_frame = 0
        self.playback_speed = 1.0
        self.is_playing = False
        self.start_time = None

    def start(self):
        """Start playback."""
        self.is_playing = True
        self.start_time = time.time()
        self.current_frame = 0

    def stop(self):
        """Stop playback."""
        self.is_playing = False

    def get_current_frame(self):
        """Get the current frame data based on elapsed time."""
        if not self.is_playing or not self.frames:
            return None

        elapsed = (time.time() - self.start_time) * self.playback_speed

        # Find the frame closest to current elapsed time
        while self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]
            if frame["t"] > elapsed:
                break
            self.current_frame += 1

        # Return current frame if available
        if self.current_frame < len(self.frames):
            return self.frames[self.current_frame]
        else:
            # Reached end of session
            self.is_playing = False
            return None

    def seek(self, time_seconds):
        """Seek to a specific time in the session."""
        for i, frame in enumerate(self.frames):
            if frame["t"] >= time_seconds:
                self.current_frame = i
                self.start_time = time.time() - (time_seconds / self.playback_speed)
                break

    def set_speed(self, speed):
        """Set playback speed (1.0 = normal, 0.5 = half speed, 2.0 = double speed)."""
        if self.is_playing:
            # Adjust start_time to maintain current position
            elapsed = (time.time() - self.start_time) * self.playback_speed
            self.start_time = time.time() - (elapsed / speed)

        self.playback_speed = speed
