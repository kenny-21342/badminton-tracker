"""Unit tests for visualizer bug fixes."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from visualizer import _speed_color, TrajectoryVisualizer


def test_speed_color_gradient():
    """Test that speed color gradient works correctly (bug fix for line 11)."""
    # Green at 0
    assert _speed_color(0) == (0, 255, 0), "Speed 0 should be green"

    # Yellow at middle (speed 20, t=0.5)
    color_20 = _speed_color(20)
    assert color_20 == (0, 255, 255), "Speed 20 should be yellow (0,255,255)"

    # Red at max (speed 40, t=1.0)
    color_40 = _speed_color(40)
    assert color_40 == (0, 0, 255), "Speed 40 should be red (0,0,255)"

    # Intermediate: green-to-yellow (speed 10, t=0.25)
    color_10 = _speed_color(10)
    assert color_10[0] == 0, "Blue channel should be 0"
    assert color_10[1] == 255, "Green channel should be 255"
    assert 0 < color_10[2] < 255, "Red channel should be between 0 and 255"

    # Intermediate: yellow-to-red (speed 30, t=0.75)
    color_30 = _speed_color(30)
    assert color_30[0] == 0, "Blue channel should be 0"
    assert 0 < color_30[1] < 255, "Green channel should be between 0 and 255"
    assert color_30[2] == 255, "Red channel should be 255"

    # Clipping at negative
    assert _speed_color(-10) == (0, 255, 0), "Negative speed should clip to green"

    # Clipping at high values
    color_100 = _speed_color(100)
    assert color_100 == (0, 0, 255), "Very high speed should be pure red"

    print("✓ All speed color gradient tests passed")


def test_visualizer_init():
    """Test that visualizer initializes with theme support."""
    vis = TrajectoryVisualizer()

    assert hasattr(vis, 'theme'), "Visualizer should have theme attribute"
    assert hasattr(vis, 'show_hud'), "Visualizer should have show_hud attribute"
    assert hasattr(vis, 'trail_opacity'), "Visualizer should have trail_opacity attribute"

    assert vis.show_minimap == True, "Minimap should be on by default"
    assert vis.show_hud == True, "HUD should be on by default"
    assert vis.trail_opacity == 0.8, "Trail opacity should be 0.8 by default"

    print("✓ Visualizer initialization tests passed")


def test_theme_switching():
    """Test that theme can be switched."""
    vis = TrajectoryVisualizer()

    # Default is dark
    assert vis.theme["bg"] == (20, 20, 20), "Default theme should be dark"

    # Switch to light
    vis.set_theme("light")
    assert vis.theme["bg"] == (240, 240, 240), "Light theme should have light bg"

    # Switch back to dark
    vis.set_theme("dark")
    assert vis.theme["bg"] == (20, 20, 20), "Dark theme should have dark bg"

    print("✓ Theme switching tests passed")


def test_division_by_zero_protection():
    """Test that draw methods handle empty trajectories safely."""
    import numpy as np
    vis = TrajectoryVisualizer()

    # Create a dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Empty trajectory should not crash
    try:
        result = vis.draw(frame, [], [], None, 0)
        assert result is not None, "Draw should return a frame"
        print("✓ Division by zero protection tests passed")
    except ZeroDivisionError:
        assert False, "Draw should not raise ZeroDivisionError on empty trajectory"


if __name__ == "__main__":
    test_speed_color_gradient()
    test_visualizer_init()
    test_theme_switching()
    test_division_by_zero_protection()
    print("\n✓ All tests passed!")
