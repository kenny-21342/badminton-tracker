"""The tracker must start without the settings panel.

imgui publishes no wheel for Python 3.14, which is this Mac's default, and its
source build fails there. Because ui_overlay imported it at module level,
`python3 main_screen.py` — the README's headline command — died with
ModuleNotFoundError before opening a window.
"""

import unittest

import numpy as np


class OptionalImguiTests(unittest.TestCase):
    def test_the_app_imports_without_imgui(self) -> None:
        import main_screen  # noqa: F401  - importing is the assertion

    def test_the_overlay_presents_its_full_surface(self) -> None:
        """Whatever OverlayUI is here, main_screen's calls must all work."""
        from ui_overlay import OverlayUI

        ui = OverlayUI()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        # Every attribute and call main_screen.py makes on it.
        returned = ui.draw(frame, None, None, None, None, "classic")
        self.assertIsNotNone(returned, "draw must return a frame to display")
        ui.toast("a message")
        ui.show_panel = not ui.show_panel
        ui.show_help = not ui.show_help


if __name__ == "__main__":
    unittest.main()
