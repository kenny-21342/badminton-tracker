import cv2
import numpy as np


class Theme:
    """Color themes for the visualizer."""
    DARK = {
        "bg": (20, 20, 20),
        "text": (240, 240, 240),
        "accent": (248, 189, 56),  # Cyan in BGR
        "panel_bg": (20, 20, 20),
        "panel_alpha": 0.9,
        "hud_bg": (0, 0, 0),
        "hud_line": (255, 255, 0),  # Yellow in BGR
    }

    LIGHT = {
        "bg": (240, 240, 240),
        "text": (20, 20, 20),
        "accent": (255, 109, 59),  # Blue in BGR
        "panel_bg": (240, 240, 240),
        "panel_alpha": 0.85,
        "hud_bg": (200, 200, 200),
        "hud_line": (0, 109, 255),  # Blue in BGR
    }


def _speed_color(speed, max_speed=40.0):
    """Map a speed to a color: slow=green, medium=yellow, fast=red."""
    t = float(np.clip(speed / max_speed, 0.0, 1.0))
    # green (0,255,0) -> yellow (0,255,255) -> red (0,0,255) in BGR
    if t < 0.5:
        k = t / 0.5
        return (0, 255, int(255 * k))
    else:
        k = (t - 0.5) / 0.5
        return (0, int(255 * (1 - k)), 255)


class TrajectoryVisualizer:
    """
    Draws the trajectory trail, the current-position marker, and a HUD.
    Coordinates passed in are expected to be in the *display* (full-frame) space.
    """

    def __init__(self):
        self.point_radius = 7
        self.show_minimap = True
        self.show_hud = True
        self.minimap_h = 120
        self.minimap_w = 214  # ~2:1 aspect, mini court
        self.theme = Theme.DARK
        self.trail_opacity = 0.8
        # "point" draws only where the shuttle is now; "trail" draws the whole
        # tracked path. Point is the default because on broadcast footage the
        # trail is a long streak across the frame, much of it extrapolated
        # through gaps, sitting on top of the thing it points at. See draw()
        # for why point mode still wants the tracker's gap filling underneath.
        self.style = "point"

    def set_theme(self, theme_name):
        """Set the color theme (dark or light)."""
        if theme_name.lower() == "dark":
            self.theme = Theme.DARK
        elif theme_name.lower() == "light":
            self.theme = Theme.LIGHT

    # ------------------------------------------------------------------- draw
    def draw(self, frame, trajectory, velocities=None, current_point=None,
             hit_flash=0, current_is_predicted=False):
        """Draw trajectory on frame. `hit_flash` > 0 paints a brief flash.

        In "point" style only the current position is drawn. On broadcast
        footage the full trail is a long streak across the frame — much of it
        extrapolated through gaps — and it reads as clutter over the very
        thing it is pointing at.

        Point style still relies on the tracker filling short gaps. Measured
        on clip_01, a marker drawn only on real TrackNet detections is present
        for 61% of in-play frames; with gap filling it is present for 86%.
        Most dropouts are one or two frames (40-80 ms) and invisible at speed,
        but the longest runs half a second, so `current_is_predicted` draws a
        hollow marker: the overlay never asserts a sighting it does not have.
        """
        if self.style == "point":
            if current_point is not None:
                self._draw_marker(frame, current_point, current_is_predicted)
            self._draw_hit_flash(frame, hit_flash)
            return frame

        overlay = frame.copy()

        # --- Trail with glow + speed coloring --------------------------------
        if len(trajectory) > 1:
            vels = velocities or [None] * len(trajectory)
            for i in range(1, len(trajectory)):
                p0 = trajectory[i - 1]
                p1 = trajectory[i]
                speed = 0.0
                if i - 1 < len(vels) and vels[i - 1] is not None:
                    speed = float(np.hypot(*vels[i - 1]))
                color = _speed_color(speed)
                alpha = i / len(trajectory)  # fade older points
                thickness = max(1, int(1 + 3 * alpha))
                # Glow pass (thick, transparent)
                cv2.line(overlay, p0, p1, color, thickness + 4)
                # Core pass
                cv2.line(overlay, p0, p1, (255, 255, 255), max(1, thickness - 1))
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            # Redraw core lines at full opacity so the trail stays visible
            for i in range(1, len(trajectory)):
                p0 = trajectory[i - 1]
                p1 = trajectory[i]
                speed = 0.0
                if i - 1 < len(vels) and vels[i - 1] is not None:
                    speed = float(np.hypot(*vels[i - 1]))
                color = _speed_color(speed)
                alpha = i / max(len(trajectory), 1)
                thickness = max(1, int(1 + 3 * alpha))
                cv2.line(frame, p0, p1, color, thickness)
        else:
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # --- Current position marker ----------------------------------------
        if current_point is not None:
            self._draw_marker(frame, current_point, current_is_predicted)

        self._draw_hit_flash(frame, hit_flash)
        return frame

    def _draw_marker(self, frame, point, is_predicted):
        """The "shuttle is here" marker.

        Hollow when the position came from the motion model rather than a
        detection, so a bridged gap is visibly a guess.
        """
        if is_predicted:
            cv2.circle(frame, point, self.point_radius + 4, (0, 0, 255), 1)
            cv2.circle(frame, point, self.point_radius, (255, 255, 255), 1)
            return
        cv2.circle(frame, point, self.point_radius + 4, (0, 0, 255), 2)
        cv2.circle(frame, point, self.point_radius, (0, 0, 255), -1)
        cv2.circle(frame, point, max(1, self.point_radius - 3),
                   (255, 255, 255), -1)

    def _draw_hit_flash(self, frame, hit_flash):
        if hit_flash <= 0:
            return
        alpha = min(0.4, hit_flash * 0.1)
        flash = np.full_like(frame, (255, 255, 255), dtype=np.uint8)
        cv2.addWeighted(flash, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, "HIT!", (frame.shape[1] // 2 - 40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

    # ------------------------------------------------------------------- HUD
    def draw_info(self, frame, trajectory_length, fps=0, rally=0, hits=0,
                  speed=0.0, extra="", rallies=None, paused=False):
        if not self.show_hud:
            return frame

        h, w = frame.shape[:2]

        # Top bar background
        cv2.rectangle(frame, (0, 0), (w, 54), self.theme["hud_bg"], -1)
        cv2.rectangle(frame, (0, 54), (w, 56), self.theme["hud_line"], -1)

        # "Rally: n" used to mean hits since the counter was last reset, which
        # read as a count of rallies. The two are now labelled for what they
        # are: hits in the rally being played, and rallies finished.
        rally_field = f"Rally: {rally} hits"
        if rallies is not None:
            rally_field += f" (#{rallies + 1})"
        info = (f"{rally_field}   Total: {hits}   "
                f"Speed: {speed:4.1f}   FPS: {fps:4.1f}   Pts: {trajectory_length}")
        if paused:
            info = "[PAUSED]   " + info
        cv2.putText(frame, info, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    self.theme["text"], 2)

        if extra:
            # Calculate actual text width to avoid overflow
            (text_width, text_height), _ = cv2.getTextSize(extra, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            available_width = w - 24  # 12px padding on each side

            if text_width > available_width - 400:  # Leave room for main info
                # Truncate with ellipsis
                chars_to_fit = int((available_width - 400) / 9)  # ~9px per char
                extra = extra[:max(0, chars_to_fit - 3)] + "..."
                (text_width, text_height), _ = cv2.getTextSize(extra, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.putText(frame, extra, (w - text_width - 12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme["accent"], 2)

        # Bottom controls bar
        cv2.rectangle(frame, (0, h - 32), (w, h), self.theme["hud_bg"], -1)

        # Split controls into multiple lines if window is narrow
        if w < 1100:
            controls_line1 = "[c] clear  [r] reset  [m] minimap  [p] point  [h] HUD  [q] quit"
            controls_line2 = "[+/-] bright  [,/.] smooth  [s] snap  [d] debug"
            cv2.rectangle(frame, (0, h - 56), (w, h), self.theme["hud_bg"], -1)
            cv2.putText(frame, controls_line1, (12, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.theme["text"], 1)
            cv2.putText(frame, controls_line2, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.theme["text"], 1)
        else:
            controls = "[c] clear  [r] reset  [m] minimap  [p] point  [+/-] bright  [,/.] smooth  [s] snap  [q] quit"
            cv2.putText(frame, controls, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.theme["text"], 1)
        return frame

    # --------------------------------------------------------------- minimap
    def draw_minimap(self, frame, trajectory, velocities=None):
        if not self.show_minimap or len(trajectory) < 2:
            return frame
        h, w = frame.shape[:2]
        mx, my = w - self.minimap_w - 14, 64
        mw, mh = self.minimap_w, self.minimap_h

        # Panel
        panel = frame.copy()
        cv2.rectangle(panel, (mx, my), (mx + mw, my + mh), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 255), 1)
        cv2.putText(frame, "Trajectory Map", (mx + 6, my + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Scale trajectory into the minimap (preserve aspect using w/h)
        sx = mw / max(w, 1)
        sy = mh / max(h, 1)
        s = min(sx, sy) * 0.9
        ox = mx + (mw - w * s) / 2
        oy = my + (mh - h * s) / 2 + 6

        vels = velocities or []
        for i in range(1, len(trajectory)):
            p0 = (int(ox + trajectory[i - 1][0] * s),
                  int(oy + trajectory[i - 1][1] * s))
            p1 = (int(ox + trajectory[i][0] * s),
                  int(oy + trajectory[i][1] * s))
            speed = 0.0
            if i - 1 < len(vels) and vels[i - 1] is not None:
                speed = float(np.hypot(*vels[i - 1]))
            cv2.line(frame, p0, p1, _speed_color(speed), 1)
        # current dot
        cx = int(ox + trajectory[-1][0] * s)
        cy = int(oy + trajectory[-1][1] * s)
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
        return frame
