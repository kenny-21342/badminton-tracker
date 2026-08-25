import cv2
import numpy as np


class ShuttlecockDetector:
    """
    Detects and tracks the shuttlecock across frames.

    Design philosophy: track-first, detect-second.
    ---------------------------------------------------------------
    Previous approach detected bright moving blobs independently each
    frame and picked the "best" one. That fails on real broadcasts
    because at brightness thresholds low enough to catch the shuttle
    (which is often blurred/dim), player jerseys and court lines also
    qualify, and players move fast enough to pass the velocity gate.

    The fix is to maintain an explicit tracking state (position +
    velocity + confidence) and use it to *predict* where the shuttle
    should be next frame, then search a small neighbourhood around
    that prediction. A jersey on the other side of the court is never
    near the prediction, so it's automatically rejected — no velocity
    heuristic needed.

    Pipeline per frame:
      1. HSV "white" + grayscale floor  -> brightness mask
      2. Background subtraction          -> motion mask
      3. AND                             -> candidate mask
      4. Extract small compact blobs     -> raw candidates
      5. If we have a confident track:
           - Predict next position from velocity.
           - Search a gate around the prediction (grows with speed
             and uncertainty).
           - If a candidate is found near the prediction, accept it.
           - If not, coast (predict-only) for a few frames; the trail
             is interpolated, not destroyed.
         If no track (cold start / lost):
           - Among all AND candidates, require minimum speed (the
             shuttle is the fastest small bright thing). This rejects
             drifting jerseys and static lines.

    All thresholds live in downscaled coordinate space.
    """

    def __init__(self, scale=0.5):
        self.scale = scale
        self.bg = None

        # --- Tunable parameters ---------------------------------------------
        self.bright_threshold = 145
        self.saturation_max = 80

        self.motion_threshold = 10
        self.bg_alpha = 0.005

        # Blob area (downscaled px²). The shuttle is small.
        self.min_area = 1
        self.max_area = 500

        # --- Tracking state -------------------------------------------------
        # The "lock" is the core anti-clutter mechanism. When we have a
        # confident track, we search only near the predicted position;
        # the rest of the frame's clutter is irrelevant.
        self.locked = False
        self.position = None          # (x, y) downscaled
        self.velocity = (0.0, 0.0)    # px/frame
        self.confidence = 0.0         # 0..1, decays on miss, grows on hit
        self.coast_frames = 0         # consecutive predict-only frames
        self.max_coast = 10           # give up lock after this many misses
        self.alpha_vel = 0.4          # velocity smoothing factor

        # Cold-start: reject slow candidates (jerseys drift, lines shimmer).
        self.cold_start_min_speed = 4.0   # downscaled px/frame
        self._prev_candidates = []        # for cold-start velocity estimation

        # Search gate around prediction.
        self.search_radius_base = 40       # downscaled px
        self.search_radius_per_speed = 1.5  # +1.5px per px/frame of speed

        # Diagnostics.
        self.debug = False
        self.last_num_candidates = 0
        self.last_stage = ""
        self.last_gate_radius = 0

    # ------------------------------------------------------------------ utils
    def _preprocess(self, frame):
        if self.scale != 1.0:
            small = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, w = gray.shape
        return gray, hsv, small, w, h

    def _white_mask(self, hsv, gray):
        h, s, v = cv2.split(hsv)
        m = cv2.bitwise_and(
            cv2.inRange(v, self.bright_threshold, 255),
            cv2.inRange(s, 0, self.saturation_max),
        )
        g = cv2.inRange(gray, self.bright_threshold, 255)
        m = cv2.bitwise_or(m, g)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return m

    def _motion_mask(self, gray):
        if self.bg is None:
            self.bg = gray.astype(np.float32)
            return None
        cv2.accumulateWeighted(gray, self.bg, self.bg_alpha)
        bg_u8 = cv2.convertScaleAbs(self.bg)
        diff = cv2.absdiff(gray, bg_u8)
        _, mask = cv2.threshold(diff, self.motion_threshold, 255,
                                cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def _extract_blobs(self, mask, w, h):
        """Return list of (cx, cy, area) for small compact blobs in mask."""
        out = []
        if mask is None:
            return out
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > 1 and ch > 1:
                aspect = max(cw, ch) / min(cw, ch)
                if aspect > 4.0:
                    continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if 1 <= cx < w - 1 and 1 <= cy < h - 1:
                out.append((cx, cy, area))
        return out

    def _predict(self):
        """Predict next position from current state."""
        if self.position is None:
            return None
        return (self.position[0] + self.velocity[0],
                self.position[1] + self.velocity[1])

    def _search_radius(self):
        speed = float(np.hypot(*self.velocity))
        # Grow with speed; also grow slightly as we coast (more uncertain).
        r = self.search_radius_base + speed * self.search_radius_per_speed
        r += self.coast_frames * 5
        self.last_gate_radius = int(r)
        return int(r)

    # ------------------------------------------------------------------ detect
    def detect(self, frame):
        gray, hsv, small, w, h = self._preprocess(frame)

        white = self._white_mask(hsv, gray)
        motion = self._motion_mask(gray)
        combined = cv2.bitwise_and(motion, white) if motion is not None else None

        # Extract candidates from the AND mask.
        candidates = self._extract_blobs(combined, w, h)
        self.last_num_candidates = len(candidates)

        result = None

        if self.locked:
            result = self._track_locked(candidates, white, w, h)
        else:
            result = self._track_coldstart(candidates, motion, white, w, h)

        self.prev_gray = gray
        return result

    # --------------------------------------------------------- locked tracking
    def _track_locked(self, candidates, white_mask, w, h):
        """
        We have a confident track. Search a small neighbourhood around
        the predicted position; ignore everything else. This is the core
        anti-jersey mechanism: a player on the far side of the court is
        never within the search gate.
        """
        pred = self._predict()
        radius = self._search_radius()

        best = None
        best_d = None
        for (cx, cy, area) in candidates:
            d = float(np.hypot(cx - pred[0], cy - pred[1]))
            if d > radius:
                continue
            score = d + abs(area - 12) * 0.05
            if best_d is None or score < best_d:
                best_d = score
                best = (cx, cy)

        if best is not None:
            self._update(best)
            self.last_stage = "track"
            return best

        # Fallback: shuttle slowed / blurred out of the motion mask but
        # is still bright. Search the white-only mask near the prediction.
        # This is safe *because* we're locked — the gate keeps us near the
        # predicted trajectory, so we can't snap to a distant jersey.
        white_candidates = self._extract_blobs(white_mask, w, h)
        for (cx, cy, area) in white_candidates:
            d = float(np.hypot(cx - pred[0], cy - pred[1]))
            if d > radius:
                continue
            score = d + abs(area - 12) * 0.05
            if best_d is None or score < best_d:
                best_d = score
                best = (cx, cy)

        if best is not None:
            self._update(best)
            self.last_stage = "track-white"
            return best
        self.coast_frames += 1
        self.confidence = max(0.0, self.confidence - 0.15)

        if self.coast_frames > self.max_coast:
            self.locked = False
            self.position = None
            self.velocity = (0.0, 0.0)
            self.last_stage = "lost"
            return None

        # Coast: return the predicted position so the tracker can
        # interpolate the gap instead of breaking the trail.
        self.last_stage = "coast"
        # Don't update velocity further during coast — keep last known.
        return (int(pred[0]), int(pred[1]))

    # ------------------------------------------------------- cold-start search
    def _track_coldstart(self, candidates, motion, white, w, h):
        """
        No confident track. We need to find the shuttle among the bright
        moving blobs. The key discriminator is SPEED: the shuttle is the
        fastest small bright object. Jerseys drift slowly, lines shimmer
        in place.

        We match each current candidate to the nearest candidate from the
        previous frame and compute the implied displacement. The candidate
        with the highest implied speed wins — as long as it's above the
        cold-start minimum (otherwise we wait, since a slow bright blob
        is more likely clutter than a shuttle).
        """
        if not candidates:
            self._prev_candidates = []
            self.last_stage = "searching"
            return None

        # If we have previous-frame candidates, estimate per-candidate speed
        # by nearest-neighbour matching.
        best = None
        best_speed = 0.0
        for (cx, cy, area) in candidates:
            if area > self.max_area * 0.5:
                continue  # skip large blobs (jerseys, body parts)
            speed = 0.0
            if self._prev_candidates:
                # Nearest previous candidate.
                nearest_d = min(
                    np.hypot(cx - px, cy - py)
                    for (px, py, _pa) in self._prev_candidates)
                speed = nearest_d  # displacement since last frame
            else:
                # No previous data — can't judge speed yet. Accept tentatively
                # but don't lock. Prefer the smallest blob.
                speed = 0.0

            if speed > best_speed:
                best_speed = speed
                best = (cx, cy)

        self._prev_candidates = list(candidates)

        if best is not None:
            # Require minimum speed to avoid locking onto drifting jerseys.
            # Exception: first frame (no prev candidates) — accept tentatively.
            if best_speed < self.cold_start_min_speed and self._prev_candidates:
                # Too slow to be a shuttle. Wait for something faster.
                self.last_stage = "wait-fast"
                return None
            self._update(best)
            # Lock as soon as we have 2 consistent detections.
            if self.confidence >= 0.4:
                self.locked = True
            self.last_stage = "acquire"
            return best

        self.last_stage = "searching"
        return None

    # ------------------------------------------------------------- state update
    def _update(self, point):
        """Update tracking state with a new detection."""
        if self.position is not None:
            new_vx = point[0] - self.position[0]
            new_vy = point[1] - self.position[1]
            a = self.alpha_vel
            self.velocity = (
                self.velocity[0] * (1 - a) + new_vx * a,
                self.velocity[1] * (1 - a) + new_vy * a,
            )
        self.position = (float(point[0]), float(point[1]))
        self.confidence = min(1.0, self.confidence + 0.25)
        self.coast_frames = 0

    # --------------------------------------------------------- public utilities
    def reset(self):
        """Clear all tracking state."""
        self.locked = False
        self.position = None
        self.velocity = (0.0, 0.0)
        self.confidence = 0.0
        self.coast_frames = 0
        self._prev_candidates = []
        self.bg = None

    def get_velocity(self):
        return self.velocity

    def get_confidence(self):
        return self.confidence

    def debug_frame(self, frame):
        """Side-by-side debug view. Marks predicted position + search gate."""
        gray, hsv, small, w, h = self._preprocess(frame)
        white = self._white_mask(hsv, gray)
        motion = self._motion_mask(gray)
        if motion is None:
            motion = np.zeros_like(gray)
        combined = cv2.bitwise_and(motion, white)

        def to_color(m):
            return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

        # Annotate the AND panel with prediction + gate + candidates.
        and_color = to_color(combined)
        if self.position is not None:
            pred = self._predict()
            radius = self._search_radius()
            # Draw search gate (cyan circle).
            cv2.circle(and_color, (int(pred[0]), int(pred[1])), radius,
                       (255, 255, 0), 1)
            # Draw predicted position (cyan dot).
            cv2.circle(and_color, (int(pred[0]), int(pred[1])), 3,
                       (255, 255, 0), -1)
            # Draw current position (green dot).
            cv2.circle(and_color, (int(self.position[0]),
                                   int(self.position[1])), 3, (0, 255, 0), -1)

        row = np.hstack([small, to_color(white), to_color(motion), and_color])
        labels = ["frame",
                  f"white v>={self.bright_threshold} s<={self.saturation_max}",
                  f"motion >={self.motion_threshold}",
                  f"AND  [gate={self.last_gate_radius} "
                  f"{'LOCKED' if self.locked else 'search'}]"]
        for i, lbl in enumerate(labels):
            cv2.putText(row, lbl, (10 + i * w, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return row
