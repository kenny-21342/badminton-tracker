#!/usr/bin/env python3
"""Check that the camera is reachable and permissions are granted.

A diagnostic you run by hand before the tracker, not part of the test suite.
It was called test_camera.py, which meant pytest collected it, opened the
camera during collection, and hit the sys.exit below — taking down the whole
run before a single real test executed.

    python3 check_camera.py
"""

import sys

import cv2


def main() -> int:
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera access failed!")
        print("\nPossible solutions:")
        print("1. Restart your Mac after modifying TCC.db")
        print("2. Check System Settings > Privacy & Security > Camera")
        print("3. Run: tccutil reset Camera (then restart)")
        return 1

    print("✅ Camera access successful!")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured: {frame.shape}")
    else:
        print("❌ Failed to capture frame")

    cap.release()
    print("\nCamera test complete. You can now run the tracker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
