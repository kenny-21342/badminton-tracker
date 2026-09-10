import math
import numpy as np
from collections import deque


class TrajectoryTracker:
    """
    Maintains the shuttlecock trajectory and provides:
      * smoothing (one-step velocity prediction + residual correction)
      * velocity-aware gap filling (when a frame has no detection)
      * hit detection (sharp direction reversal OR big speed change)
      * rally counting
      * out-of-frame freeze: when the shuttle leaves the frame, the trail
        is frozen (not extrapolated into empty space). When it reappears,
        tracking resumes.
      * camera-cut detection: when the scene changes abruptly (director
        switches camera), the trail is auto-cleared so a stale path from
        the previous angle doesn't linger.

    All coordinates are in the space the detector emits (downscaled for
    classic CV, full-frame for YOLO).
    """

    def __init__(self, max_trajectory_length=60, hit_detection_threshold=25,
                 smoothing=0.5, cut_threshold=40.0,
                 adaptive_smoothing=True):
        self.trajectory = deque(maxlen=max_trajectory_length)
        self.point_is_predicted = deque(maxlen=max_trajectory_length)
        self.velocities = deque(maxlen=max_trajectory_length)
        self.hit_detection_threshold = hit_detection_threshold
        self.smoothing = smoothing  # 0 = trust detection, 1 = trust prediction
        # With adaptive smoothing on, `smoothing` is a ceiling rather than a
        # fixed blend: see _effective_smoothing for why a fixed one costs
        # accuracy without buying much steadiness.
        self.adaptive_smoothing = adaptive_smoothing
        self.frames_since_acquired = 0
        self.no_detection_count = 0
        self.max_no_detection = 12   # frames to keep predicting before giving up
        self.last_velocity = (0.0, 0.0)
        self.last_smoothed = None    # last smoothed position
        self.rally_count = 0         # hits in the current rally
        self.hit_count = 0           # hits this session
        self.cooldown = 0            # frames to wait after a hit

        # Rallies. A rally ends when the shuttle stays gone long enough that
        # the point must be over, and the next detection begins the next one.
        # Previously "rally" was incremented on the same line as hit_count and
        # only differed because a keypress zeroed it, so the HUD's "rally"
        # field was really "hits since you last pressed R".
        self.rallies_completed = 0
        self.rally_lengths = []      # hits per completed rally
        self.rally_active = False

        # Hit detection compares velocities in pixels per frame, so the same
        # stroke reads differently at 12 fps than at 40 — and this pipeline
        # ranges from ~2 fps (TrackNet on CPU) to ~170 (classic CV). The
        # threshold is therefore scaled against a reference rate, and the
        # cooldown and gap limits are held in seconds rather than frames.
        # Vertical acceleration, px/frame^2, estimated from real detections.
        # yolo_detector already had this and a parabolic predictor whose
        # docstring called it "what makes the gap-fill follow cross-court
        # smashes instead of predicting straight lines" — but that function was
        # never called, and gap-filling happens here, for every detector. So a
        # shuttle that vanished mid-flight was coasted along a straight line
        # while the real one was falling away from it.
        self.gravity_px = 0.0

        self.reference_fps = 30.0
        self.observed_fps = 30.0
        self.hit_cooldown_seconds = 0.4
        self.max_no_detection_seconds = 0.4

        # Out-of-frame handling.
        self.frame_width = None
        self.frame_height = None
        self.edge_margin = 10        # px tolerance for "near edge"
        self.out_of_frame = False    # shuttle has left the visible area

        # Camera-cut detection.
        self.cut_threshold = cut_threshold  # mean-frame-diff that signals a cut
        self._prev_thumb = None       # previous frame thumbnail for diffing

    # ------------------------------------------------------------- public API
    def set_frame_size(self, width, height):
        self.frame_width = width
        self.frame_height = height

    def _is_near_edge(self, point):
        """True if the shuttle is near or beyond the frame border."""
        if self.frame_width is None or self.frame_height is None:
            return False
        m = self.edge_margin
        x, y = point
        return (x < m or x >= self.frame_width - m or
                y < m or y >= self.frame_height - m)

    def _is_off_frame(self, point):
        """True if the predicted position is outside the frame."""
        if self.frame_width is None or self.frame_height is None:
            return False
        x, y = point
        return (x < 0 or x >= self.frame_width or
                y < 0 or y >= self.frame_height)

    def check_camera_cut(self, frame):
        """
        Detect an abrupt scene change (camera switch, replay transition).
        Uses a tiny grayscale thumbnail mean-absolute-difference. Returns
        True if the frame looks like a cut from the previous one.
        """
        thumb = cv2_resize_gray(frame, 32)
        if self._prev_thumb is None:
            self._prev_thumb = thumb
            return False
        diff = float(np.mean(np.abs(thumb.astype(np.int16) -
                                     self._prev_thumb.astype(np.int16))))
        self._prev_thumb = thumb
        return diff > self.cut_threshold

    def add_point(self, point, frame=None):
        """
        Add a detected point (or None). Returns True if a hit was detected.

        Args:
            point: (x, y) or None.
            frame: the current video frame (for camera-cut detection).
                   Pass None to skip cut detection.

        On gaps (point is None), we predict forward to bridge short gaps
        — UNLESS the shuttle was near the frame edge (out of frame), in
        which case we freeze the trail until it reappears.
        """
        # Camera-cut detection: auto-clear on scene change.
        if frame is not None and self.check_camera_cut(frame):
            if len(self.trajectory) > 3:
                self.clear_trajectory()
                return False

        if point is None:
            self.no_detection_count += 1
            self._maybe_end_rally()

            # Predict where the shuttle would be next frame.
            predicted = None
            if (self.last_smoothed is not None
                    and self.last_velocity != (0.0, 0.0)):
                predicted, predicted_velocity = self._predict_from(
                    self.last_smoothed, self.last_velocity)

            # If the predicted position is off-frame, the shuttle has left
            # the visible area. Freeze the trail — don't coast into space.
            if predicted is not None and self._is_off_frame(predicted):
                self.out_of_frame = True
                return False

            # If it was already out of frame, stay frozen.
            if self.out_of_frame:
                return False

            # Normal gap (occlusion, blur): predict forward to bridge it.
            if (self.no_detection_count <= self.max_no_detection
                    and predicted is not None):
                self._append_smoothed(predicted, predicted, is_predicted=True,
                                      vel=predicted_velocity)
            # Gap too big: stop extending but preserve existing trail.
            return False

        # We have a real detection.
        # Check if it re-entered from off-frame.
        if self.out_of_frame:
            self.out_of_frame = False
            # Reset velocity so we don't smear from the old off-frame position.
            self.last_velocity = (0.0, 0.0)
            # The velocity estimate starts again, so the track is young again.
            self.frames_since_acquired = 0

        # Check if it's now near the edge → mark as potentially leaving.
        if self._is_near_edge(point):
            self.out_of_frame = True

        self.no_detection_count = 0
        if not self.rally_active:
            # First sight of the shuttle since the last rally ended.
            self.rally_active = True
            self.rally_count = 0
        self.frames_since_acquired += 1
        if self.last_smoothed is None:
            smoothed = point
            vel = (0.0, 0.0)
        else:
            predicted, _ = self._predict_from(self.last_smoothed,
                                              self.last_velocity)
            s = self._effective_smoothing(predicted, point)
            smoothed = (
                predicted[0] * s + point[0] * (1 - s),
                predicted[1] * s + point[1] * (1 - s),
            )
            vel = (smoothed[0] - self.last_smoothed[0],
                   smoothed[1] - self.last_smoothed[1])

        hit = self._append_smoothed(smoothed, point, is_predicted=False, vel=vel)
        return hit

    # ------------------------------------------------------------- internals
    # How far the prediction may miss before it stops being trusted. Roughly
    # the detector's own median error (4 px on the labelled clips), so a
    # prediction no worse than the detector keeps full weight.
    _RESIDUAL_TOLERANCE = 6.0
    # Frames of real detections needed before the velocity estimate is worth
    # blending towards at all.
    _MATURITY_FRAMES = 4

    def _effective_smoothing(self, predicted, point):
        """How far to lean on the prediction for this frame, in 0..1.

        A fixed blend is the wrong shape for a shuttlecock. Measured against
        the labelled clips in datasets/eval_ld2011, holding TrackNet fixed and
        varying only this number, a fixed blend costs accuracy at every
        setting and buys very little steadiness:

            s     median error   mean jerk
            0.0       4.00 px      25.0
            0.1       4.53 px      19.8
            0.5       9.01 px      14.7      <- the old default
            0.9     161.00 px      23.7

        The reason is lag. A one-step linear-plus-gravity prediction cannot
        keep up with a smash, so blending towards it drags the point behind
        the shuttle; past s=0.6 the prediction diverges and the trail is not
        even smooth any more.

        So the blend is spent only where it is cheap. Two things pull it down:

        - Disagreement. When the detection lands far from the prediction the
          motion model has been overtaken by events -- a hit, a direction
          change, a fast rally -- and the detection is the better estimate.
        - Youth. Just after the shuttle is picked up the velocity estimate is
          built from one or two frames and predicts nearly nothing useful.

        When neither applies -- a slow, well-tracked shuttle whose prediction
        lands on top of the detection -- smoothing is close to free, and that
        is the case it is allowed to act on.
        """
        if not self.adaptive_smoothing:
            return self.smoothing

        maturity = min(1.0, self.frames_since_acquired / self._MATURITY_FRAMES)

        residual = math.hypot(predicted[0] - point[0], predicted[1] - point[1])
        # Falls off smoothly rather than at a threshold, so a trail does not
        # visibly change character frame to frame.
        agreement = 1.0 / (1.0 + (residual / self._RESIDUAL_TOLERANCE) ** 2)

        return self.smoothing * maturity * agreement

    def _append_smoothed(self, smoothed, raw, is_predicted, vel=None):
        pt = (int(smoothed[0]), int(smoothed[1]))
        self.trajectory.append(pt)
        # A predicted point is the motion model talking to itself. Feeding it
        # back into hit detection lets the tracker invent a direction reversal
        # it never saw, so the flags are kept and the test skips over them.
        self.point_is_predicted.append(bool(is_predicted))
        self.last_smoothed = (smoothed[0], smoothed[1])

        if vel is None:
            if len(self.trajectory) >= 2:
                a, b = self.trajectory[-2], self.trajectory[-1]
                vel = (float(b[0] - a[0]), float(b[1] - a[1]))
            else:
                vel = (0.0, 0.0)
        self.velocities.append(vel)
        self.last_velocity = vel
        if not is_predicted:
            self._estimate_gravity()

        if self.cooldown > 0:
            self.cooldown -= 1

        if not is_predicted and self.cooldown == 0 and len(self.trajectory) >= 6:
            if self._detect_hit():
                self.hit_count += 1
                self.rally_count += 1
                self.cooldown = max(1, round(self.hit_cooldown_seconds
                                             * self.observed_fps))
                return True
        return False

    def _predict_from(self, position, velocity):
        """One step of a parabola, returning the new position and velocity.

        Both are returned because deriving the velocity back out of the
        positions double-counts the acceleration: the step already contains
        half of it, so adding a full g again over-accelerates by g/2 every
        frame and compounds over a gap.

            y' = y + v + g/2      v' = v + g
        """
        new_velocity = (velocity[0], velocity[1] + self.gravity_px)
        new_position = (
            position[0] + velocity[0],
            position[1] + velocity[1] + 0.5 * self.gravity_px,
        )
        return new_position, new_velocity

    def _estimate_gravity(self):
        """Second derivative of vertical position over recent real points.

        Predicted points are excluded: they were produced by this estimate, so
        including them would feed it back into itself and drift.
        """
        real = [
            point
            for point, predicted in zip(self.trajectory, self.point_is_predicted)
            if not predicted
        ]
        if len(real) < 5:
            return
        recent = real[-5:]
        velocities = [recent[i + 1][1] - recent[i][1] for i in range(len(recent) - 1)]
        accelerations = [
            velocities[i + 1] - velocities[i] for i in range(len(velocities) - 1)
        ]
        if not accelerations:
            return
        measured = sum(accelerations) / len(accelerations)
        # Smoothed: a single noisy detection should not swing the estimate.
        self.gravity_px = self.gravity_px * 0.7 + measured * 0.3

    def end_rally(self):
        """Close the current rally now, whatever the detector is seeing.

        Used when the operator pauses: whatever is on screen during a replay
        or an advert is not the rally continuing, and resuming later should
        not append to it.
        """
        if not self.rally_active:
            return
        self.rally_active = False
        if self.rally_count > 0:
            self.rallies_completed += 1
            self.rally_lengths.append(self.rally_count)

    def _maybe_end_rally(self):
        """Close the rally once the shuttle has been gone long enough.

        The shuttle vanishing for a moment is an occlusion; vanishing for
        longer means the point is over. The same limit already governs how
        long the tracker will keep predicting forward.
        """
        if not self.rally_active:
            return
        limit = max(1, round(self.max_no_detection_seconds * self.observed_fps))
        if self.no_detection_count <= limit:
            return
        self.rally_active = False
        if self.rally_count > 0:
            self.rallies_completed += 1
            self.rally_lengths.append(self.rally_count)


    def _detect_hit(self):
        """
        A hit = sharp reversal in vertical OR horizontal velocity, or a large
        deceleration. We compare the mean velocity of the last few points
        against the mean of the few before them.
        """
        # Only real detections. A predicted point is an extrapolation of the
        # last velocity, so a segment containing one always agrees with the
        # motion model and can hide a genuine reversal — or, after a bad
        # prediction, manufacture one.
        pts = [
            point
            for point, predicted in zip(self.trajectory, self.point_is_predicted)
            if not predicted
        ]
        n = len(pts)
        if n < 6:
            return False

        recent = pts[-3:]
        prior = pts[-6:-3]

        def mean_vel(seg):
            dxs, dys = [], []
            for i in range(len(seg) - 1):
                dxs.append(seg[i + 1][0] - seg[i][0])
                dys.append(seg[i + 1][1] - seg[i][1])
            return (sum(dxs) / max(len(dxs), 1), sum(dys) / max(len(dys), 1))

        v1 = mean_vel(prior)   # older segment
        v2 = mean_vel(recent)  # newer segment

        speed1 = math.hypot(*v1)
        speed2 = math.hypot(*v2)

        # These velocities are pixels per frame, so at 12 fps the shuttle moves
        # two and a half times as far between samples as at 30 for the same
        # stroke. Scaling the threshold keeps a hit the same physical event
        # whichever detector is running.
        threshold = self.hit_detection_threshold * (
            self.reference_fps / max(self.observed_fps, 1.0)
        )

        # Reversal of vertical direction (most common badminton hit signature)
        if v1[1] * v2[1] < 0 and abs(v1[1] - v2[1]) > threshold:
            return True
        # Reversal of horizontal direction
        if v1[0] * v2[0] < 0 and abs(v1[0] - v2[0]) > threshold:
            return True
        # Big deceleration (net shot / drop)
        if speed1 > threshold and speed2 < speed1 * 0.25:
            return True
        return False

    # ------------------------------------------------------------- utilities
    def clear_trajectory(self):
        self.trajectory.clear()
        self.point_is_predicted.clear()
        self.gravity_px = 0.0
        self.velocities.clear()
        self.last_velocity = (0.0, 0.0)
        self.last_smoothed = None
        self.no_detection_count = 0
        self.frames_since_acquired = 0
        self.cooldown = 0
        self.out_of_frame = False

    def reset_rally(self):
        self.rally_count = 0

    def get_trajectory(self):
        return list(self.trajectory)

    def get_velocities(self):
        return list(self.velocities)

    def current_speed(self):
        return math.hypot(*self.last_velocity) if self.last_velocity else 0.0


# --------------------------------------------------------------- module helpers
def cv2_resize_gray(frame, size=32):
    """Downscale a frame to a tiny grayscale thumbnail for fast comparison."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
