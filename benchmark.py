#!/usr/bin/env python3
"""
How fast each detector runs, without a screen or a disk in the way.

    python3 benchmark.py
    python3 benchmark.py --frames 30

Detection speed and detection *accuracy* are separate questions and want
separate tools: evaluate.py answers the second by reading labelled JPEGs, and
its fps column is dominated by JPEG decode and by resetting the detector at
every cut. Taking that column for a detector's speed is how TrackNet came to
be recorded at 2 fps when it does 12 at the same settings, and 42 at reduced
resolution.

So this feeds synthetic frames straight to the detector: no capture, no
decode, no resets. The result is an upper bound on what the live pipeline can
reach, which is the number worth knowing when choosing a resolution.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FRAME_SIZE = (720, 1280, 3)   # what a typical screen capture hands over


def timed(detector, frames: int) -> tuple[float, float]:
    """Return (fps, ms per frame), after a warm-up.

    The warm-up matters: TrackNet reads three consecutive frames, so its first
    two calls are a cold start that would otherwise drag the average down.
    """
    frame = (np.random.rand(*FRAME_SIZE) * 255).astype(np.uint8)
    for _ in range(3):
        detector.detect(frame)

    started = time.time()
    for _ in range(frames):
        detector.detect(frame)
    elapsed = time.time() - started
    return frames / elapsed, 1000 * elapsed / frames


def configurations():
    """Each entry is (label, build) so a failure names what could not load."""
    def classic():
        from detector import ShuttlecockDetector

        return ShuttlecockDetector()

    def classic_full():
        from detector import ShuttlecockDetector

        return ShuttlecockDetector(scale=1.0)

    def tracknet(width, height, fp16):
        def build():
            from tracknet_detector import TrackNetDetector

            weights = ROOT / "models" / "TrackNet_best.pt"
            if not weights.exists():
                raise FileNotFoundError(
                    "models/TrackNet_best.pt is missing — "
                    "run python3 download_tracknet.py"
                )
            return TrackNetDetector(
                str(weights), input_width=width, input_height=height, fp16=fp16
            )

        return build

    def yolo():
        from yolo_detector import YOLODetector

        return YOLODetector()

    return [
        ("classic CV (scale 0.5)", classic),
        ("classic CV (full res)", classic_full),
        ("tracknet 512x288 fp32", tracknet(512, 288, False)),
        ("tracknet 512x288 fp16", tracknet(512, 288, True)),
        ("tracknet 384x216 fp16", tracknet(384, 216, True)),
        ("tracknet 256x144 fp16", tracknet(256, 144, True)),
        ("yolo", yolo),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=15,
                        help="timed frames per configuration")
    args = parser.parse_args()

    print(f"{FRAME_SIZE[1]}x{FRAME_SIZE[0]} synthetic frames, "
          f"{args.frames} timed per configuration\n")
    print(f"{'detector':24} {'fps':>7} {'ms/frame':>10}  device")
    print("-" * 60)

    for label, build in configurations():
        try:
            detector = build()
        except Exception as error:
            print(f"{label:24} {'—':>7} {'—':>10}  {type(error).__name__}: "
                  f"{str(error)[:40]}")
            continue
        device = getattr(detector, "device", "cpu")
        fps, milliseconds = timed(detector, args.frames)
        print(f"{label:24} {fps:7.1f} {milliseconds:10.1f}  {device}")

    print("\nUpper bounds: no screen capture, no drawing, no disk. The live "
          "pipeline\nadds capture and overlay on top of these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
