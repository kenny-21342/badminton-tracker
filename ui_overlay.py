import cv2
import numpy as np
import time

# The settings panel is a convenience, not the product — the tracker captures,
# detects and draws without it. imgui publishes no wheel for Python 3.14 (this
# Mac's default) and its source build fails, which made `import imgui` here
# stop main_screen.py from starting at all: the README's headline command died
# with ModuleNotFoundError before opening a window.
try:
    import imgui
    from imgui.integrations.opencv import OpenCVRenderer

    IMGUI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the interpreter
    imgui = None
    OpenCVRenderer = None
    IMGUI_AVAILABLE = False


class OverlayUI:
    """
    ImGui-based overlay UI for real-time parameter adjustment.
    Renders semi-transparent panels directly onto the OpenCV frame.
    """

    def __init__(self):
        self.show_panel = False  # Toggle with TAB
        self.show_help = False   # Toggle with ?
        self.toast_queue = []    # [(message, end_time), ...]
        self.current_tab = 0     # Active settings tab

        # Initialize ImGui
        imgui.create_context()
        self.renderer = None  # Will be initialized on first draw
        self.window_flags = (
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_COLLAPSE
        )

        # Theme colors
        self.theme = "dark"
        self._apply_theme()

    def _apply_theme(self):
        """Apply color theme to ImGui."""
        style = imgui.get_style()

        if self.theme == "dark":
            # Dark theme with cyan accent
            imgui.style_colors_dark()
            style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.08, 0.08, 0.08, 0.85)
            style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.14, 0.59, 0.97, 0.9)
            style.colors[imgui.COLOR_BUTTON] = (0.14, 0.59, 0.97, 0.6)
            style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.14, 0.59, 0.97, 0.8)
            style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.14, 0.59, 0.97, 1.0)
            style.colors[imgui.COLOR_SLIDER_GRAB] = (0.14, 0.59, 0.97, 0.8)
            style.colors[imgui.COLOR_SLIDER_GRAB_ACTIVE] = (0.14, 0.59, 0.97, 1.0)
        else:
            # Light theme with blue accent
            imgui.style_colors_light()
            style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.94, 0.94, 0.94, 0.85)
            style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.23, 0.43, 1.0, 0.9)

        style.window_rounding = 8.0
        style.frame_rounding = 4.0
        style.grab_rounding = 4.0

    def draw(self, frame, detector, tracker, visualizer, fps=0):
        """
        Main draw function. Renders ImGui overlay onto OpenCV frame.
        Returns modified frame with UI drawn.
        """
        h, w = frame.shape[:2]

        # Initialize renderer on first call
        if self.renderer is None:
            self.renderer = OpenCVRenderer()

        # Start new ImGui frame
        imgui.new_frame()

        # Draw settings panel if visible
        if self.show_panel:
            self._draw_settings_panel(detector, tracker, visualizer, fps)

        # Draw help overlay if visible
        if self.show_help:
            self._draw_help_overlay(w, h)

        # Draw toast notifications
        self._draw_toasts(w, h)

        # Draw detector status badge
        self._draw_status_badge(detector, tracker)

        # Render ImGui to frame
        imgui.render()
        self.renderer.render(imgui.get_draw_data())

        return frame

    def _draw_settings_panel(self, detector, tracker, visualizer, fps):
        """Main settings panel with tabs."""
        imgui.set_next_window_position(20, 80)
        imgui.set_next_window_size(380, 500)

        imgui.begin("Settings", flags=self.window_flags)

        # Tab bar
        if imgui.begin_tab_bar("SettingsTabs"):
            if imgui.begin_tab_item("Detection").selected:
                self._tab_detection(detector)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Tracking").selected:
                self._tab_tracking(tracker)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Display").selected:
                self._tab_display(visualizer)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Session").selected:
                self._tab_session(fps)
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()

    def _tab_detection(self, detector):
        """Detection settings tab."""
        imgui.text("Detector Configuration")
        imgui.separator()
        imgui.spacing()

        # Detector mode dropdown
        detector_type = type(detector).__name__
        imgui.text(f"Current Mode: {detector_type}")
        imgui.spacing()

        # Show detector-specific settings
        if hasattr(detector, 'brightness_threshold'):
            # Classic CV detector
            imgui.text("Classic CV Settings")
            changed, value = imgui.slider_int(
                "Brightness", detector.brightness_threshold, 0, 255
            )
            if changed:
                detector.brightness_threshold = value
                self.toast("Brightness updated")

        elif hasattr(detector, 'confidence_threshold'):
            # YOLO or TrackNet detector
            imgui.text("AI Detector Settings")
            changed, value = imgui.slider_float(
                "Confidence", detector.confidence_threshold, 0.0, 1.0
            )
            if changed:
                detector.confidence_threshold = value
                self.toast(f"Confidence: {value:.2f}")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if imgui.button("Reset Detector", width=200):
            detector.reset()
            self.toast("Detector reset")

    def _tab_tracking(self, tracker):
        """Tracking settings tab."""
        imgui.text("Trajectory Tracking")
        imgui.separator()
        imgui.spacing()

        # Smoothing slider
        changed, value = imgui.slider_float(
            "Smoothing", tracker.smoothing, 0.0, 0.95, "%.2f"
        )
        if changed:
            tracker.smoothing = value
            self.toast(f"Smoothing: {value:.2f}")

        imgui.spacing()

        # Hit detection threshold
        if hasattr(tracker, 'hit_threshold'):
            changed, value = imgui.slider_int(
                "Hit Threshold", tracker.hit_threshold, 10, 50
            )
            if changed:
                tracker.hit_threshold = value
                self.toast(f"Hit threshold: {value}")

        imgui.spacing()

        # Rally cooldown
        if hasattr(tracker, 'rally_cooldown'):
            changed, value = imgui.slider_int(
                "Rally Cooldown", tracker.rally_cooldown, 5, 20
            )
            if changed:
                tracker.rally_cooldown = value

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Action buttons
        if imgui.button("Clear Trajectory", width=200):
            tracker.trajectory.clear()
            tracker.velocities.clear()
            self.toast("Trajectory cleared")

        imgui.spacing()

        if imgui.button("Reset Rally Count", width=200):
            tracker.rally_count = 0
            tracker.hit_count = 0
            self.toast("Rally reset")

    def _tab_display(self, visualizer):
        """Display settings tab."""
        imgui.text("Visual Settings")
        imgui.separator()
        imgui.spacing()

        # Show/hide toggles
        changed, value = imgui.checkbox("Show HUD", visualizer.show_hud)
        if changed:
            visualizer.show_hud = value

        imgui.spacing()

        changed, value = imgui.checkbox("Show Minimap", visualizer.show_minimap)
        if changed:
            visualizer.show_minimap = value
            self.toast("Minimap toggled")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Theme selector
        imgui.text("Theme")
        if imgui.radio_button("Dark", self.theme == "dark"):
            self.theme = "dark"
            self._apply_theme()
            self.toast("Dark theme")

        imgui.same_line()

        if imgui.radio_button("Light", self.theme == "light"):
            self.theme = "light"
            self._apply_theme()
            self.toast("Light theme")

    def _tab_session(self, fps):
        """Session recording tab."""
        imgui.text("Session Info")
        imgui.separator()
        imgui.spacing()

        # Display session stats (placeholder)
        imgui.text(f"FPS: {fps:.1f}")
        imgui.text("Duration: --:--:--")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Recording controls (placeholder)
        imgui.text("Recording: Not implemented")
        if imgui.button("Record", width=90):
            self.toast("Recording feature coming soon")

        imgui.same_line()

        if imgui.button("Export CSV", width=90):
            self.toast("Export feature coming soon")

    def _draw_help_overlay(self, w, h):
        """Draw keyboard shortcuts help overlay."""
        imgui.set_next_window_position(w // 2 - 200, h // 2 - 200)
        imgui.set_next_window_size(400, 400)

        imgui.begin("Keyboard Shortcuts", flags=self.window_flags)

        imgui.text("Controls")
        imgui.separator()
        imgui.spacing()

        shortcuts = [
            ("TAB", "Toggle settings panel"),
            ("? or F1", "Toggle this help"),
            ("c", "Clear trajectory"),
            ("r", "Reset rally count"),
            ("m", "Toggle minimap"),
            ("h", "Toggle HUD"),
            ("+/-", "Adjust brightness"),
            (",/.", "Adjust smoothing"),
            ("d", "Toggle debug info"),
            ("q", "Quit"),
        ]

        for key, desc in shortcuts:
            imgui.text(f"{key:10s}  {desc}")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if imgui.button("Close", width=100):
            self.show_help = False

        imgui.end()

    def _draw_toasts(self, w, h):
        """Draw toast notifications at top-center."""
        current_time = time.time()
        # Remove expired toasts
        self.toast_queue = [(msg, end_time) for msg, end_time in self.toast_queue
                           if end_time > current_time]

        if not self.toast_queue:
            return

        # Draw active toasts
        y_offset = 80
        for msg, end_time in self.toast_queue:
            # Fade out in last 0.5 seconds
            time_left = end_time - current_time
            alpha = min(1.0, time_left / 0.5)

            imgui.set_next_window_position(w // 2 - 100, y_offset)
            imgui.set_next_window_bg_alpha(alpha * 0.8)

            imgui.begin(f"##toast_{id(msg)}",
                       flags=imgui.WINDOW_NO_TITLE_BAR |
                             imgui.WINDOW_NO_RESIZE |
                             imgui.WINDOW_NO_MOVE |
                             imgui.WINDOW_NO_SCROLLBAR)
            imgui.text(msg)
            imgui.end()

            y_offset += 40

    def _draw_status_badge(self, detector, tracker):
        """Draw detector status badge (locked/coasting/lost)."""
        # Determine status
        if tracker.lost_count == 0:
            status = "LOCKED"
            color = (0.2, 0.8, 0.2, 0.9)  # Green
        elif tracker.lost_count < tracker.max_lost:
            status = "COAST"
            color = (0.9, 0.7, 0.2, 0.9)  # Yellow
        else:
            status = "LOST"
            color = (0.9, 0.2, 0.2, 0.9)  # Red

        imgui.set_next_window_position(20, 70)
        imgui.set_next_window_bg_alpha(0.7)

        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *color)
        imgui.begin("##status",
                   flags=imgui.WINDOW_NO_TITLE_BAR |
                         imgui.WINDOW_NO_RESIZE |
                         imgui.WINDOW_NO_MOVE |
                         imgui.WINDOW_NO_SCROLLBAR)
        imgui.text(f"● {status}")
        imgui.end()
        imgui.pop_style_color()

    def toast(self, message, duration=2.0):
        """Queue a toast notification."""
        end_time = time.time() + duration
        self.toast_queue.append((message, end_time))

    def toggle_panel(self):
        """Toggle settings panel visibility."""
        self.show_panel = not self.show_panel

    def toggle_help(self):
        """Toggle help overlay visibility."""
        self.show_help = not self.show_help


class _NoOverlay:
    """Stand-in used when imgui is unavailable.

    Presents the same surface main_screen.py calls — draw, toast, show_panel,
    show_help — so the tracker needs no conditionals and simply runs without a
    settings panel. Keyboard shortcuts that change settings still work; only
    the panel that displays them is missing.
    """

    def __init__(self) -> None:
        self.show_panel = False
        self.show_help = False
        self._warned = False

    def draw(self, frame, *args, **kwargs):
        return frame

    def toast(self, message: str) -> None:
        # Toasts are the only feedback for some shortcuts, so without a panel
        # they go to the terminal rather than vanishing.
        print(f"[tracker] {message}", flush=True)

    def __getattr__(self, name):
        # Anything else the UI grows later degrades to a no-op rather than
        # crashing a run that is otherwise working.
        def _noop(*args, **kwargs):
            return None

        return _noop


if not IMGUI_AVAILABLE:
    _RealOverlayUI = OverlayUI

    class OverlayUI(_NoOverlay):  # type: ignore[no-redef]
        def __init__(self) -> None:
            super().__init__()
            print(
                "[tracker] imgui is not installed, so the TAB settings panel is "
                "unavailable. Everything else works; `pip install imgui[opencv]` "
                "to enable it (needs Python <= 3.13).",
                flush=True,
            )
