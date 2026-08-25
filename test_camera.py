#!/usr/bin/env python3
"""Simple camera test to verify permissions."""

import cv2
import sys

print("Testing camera access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera access failed!")
    print("\nPossible solutions:")
    print("1. Restart your Mac after modifying TCC.db")
    print("2. Check System Settings > Privacy & Security > Camera")
    print("3. Run: tccutil reset Camera (then restart)")
    sys.exit(1)

print("✅ Camera access successful!")
ret, frame = cap.read()
if ret:
    print(f"✅ Frame captured: {frame.shape}")
else:
    print("❌ Failed to capture frame")

cap.release()
print("\nCamera test complete. You can now run the tracker.")
