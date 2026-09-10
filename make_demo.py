"""
Render an annotated demo clip from a video file, headlessly.

The tracker itself reads the screen, which makes a demo awkward to capture: it
needs a video playing full-screen, the permission granted, and a screen
recorder running over the top. This renders the same overlay straight from a
video file instead, so the clip is reproducible and does not depend on what
was on screen at the time.

    python3 make_demo.py --input match.mp4 --start 1830 --duration 12 \
        --output demo.mp4

The detector, tracker and visualiser are the ones main_screen.py uses, so what
comes out is what the app draws -- not a separate rendering path that could
drift from it.
"""
import argparse
import sys
import time

import cv2

from tracker import TrajectoryTracker
from visualizer import TrajectoryVisualizer


def unwrap(point):
    """Detectors return either (x, y) or ((x, y), confidence)."""
    if point is None:
        return None
    if isinstance(point[0], (int, float)):
        return point
    return point[0]


def build_detector(args):
    """TrackNet unless --cv, matching main_screen.py's measured default."""
    if args.cv:
        from detector import ShuttlecockDetector

        return ShuttlecockDetector(), 1.0 / ShuttlecockDetector().scale
    from tracknet_detector import TrackNetDetector

    return TrackNetDetector(args.model), 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Source video file.")
    ap.add_argument("--output", default="demo.mp4", help="Annotated output.")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Start time in seconds.")
    ap.add_argument("--duration", type=float, default=12.0,
                    help="Clip length in seconds.")
    ap.add_argument("--cv", action="store_true",
                    help="Use the classic CV detector instead of TrackNet.")
    ap.add_argument("--model", default=None, help="TrackNet weights path.")
    ap.add_argument("--trail", type=int, default=30,
                    help="Trail length in points.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open {args.input}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = int(args.start * fps)
    count = int(args.duration * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    detector, inv_scale = build_detector(args)
    tracker = TrajectoryTracker(max_trajectory_length=args.trail,
                                hit_detection_threshold=25)
    tracker.set_frame_size(width, height)
    # The source is a fixed-rate file, so the frame-rate-dependent thresholds
    # can be told the truth instead of estimating it from wall-clock time.
    tracker.observed_fps = fps
    visualizer = TrajectoryVisualizer()

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        print(f"Error: cannot write {args.output}", file=sys.stderr)
        return 1

    print(f"{args.input}: {width}x{height} @ {fps:g} fps")
    print(f"rendering {args.duration:g}s from {args.start:g}s "
          f"({count} frames) with "
          f"{'classic CV' if args.cv else 'TrackNet'}")

    began = time.time()
    detected = 0
    for index in range(count):
        ok, frame = cap.read()
        if not ok:
            print(f"source ended after {index} frames")
            break

        point = unwrap(detector.detect(frame))
        # Camera cuts matter here in a way they do not on a live stream: a
        # broadcast replay would otherwise drag the previous angle's trail
        # across the new one.
        tracker.add_point(point, frame=frame)
        if point is not None:
            detected += 1

        trajectory = [(int(x * inv_scale), int(y * inv_scale))
                      for x, y in tracker.get_trajectory()]
        current = trajectory[-1] if trajectory else None
        visualizer.draw(frame, trajectory, tracker.get_velocities(),
                        current_point=current)
        visualizer.draw_info(frame, len(trajectory), fps=fps,
                             rally=tracker.rally_count,
                             hits=tracker.hit_count,
                             speed=tracker.current_speed() * inv_scale,
                             rallies=tracker.rallies_completed)
        writer.write(frame)

        if index and index % 25 == 0:
            print(f"  {index}/{count} frames, {detected} detections",
                  end="\r", flush=True)

    writer.release()
    cap.release()
    elapsed = time.time() - began
    print(f"\nwrote {args.output}: {detected} detections, "
          f"{tracker.hit_count} hits, {tracker.rallies_completed} rallies "
          f"completed, in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
