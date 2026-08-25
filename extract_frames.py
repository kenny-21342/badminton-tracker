"""
Extract frames for labeling.

Samples frames from a video file or live screen capture, spread evenly
throughout, so you can label a diverse set of shuttlecock positions
(fast flight, slow drop, near net, near baseline, etc.).

Frame classifier:
  Badminton broadcasts include player close-ups, crowd shots, slow-motion
  replays, Hawk-Eye 3D graphics, and title cards — all useless for shuttle
  labeling. The frame_classifier module analyzes each frame and keeps only
  genuine live court footage.

Usage:
    # From a video file (e.g. a recorded match):
    python3 extract_frames.py --video match.mp4 --count 200

    # Disable classifier (extract every Nth frame regardless):
    python3 extract_frames.py --video match.mp4 --count 200 --no-filter

    # Show what gets filtered (saves rejected frames to a debug folder):
    python3 extract_frames.py --video match.mp4 --count 200 --show-rejected

    # From live screen capture (grab N screenshots with 2s spacing):
    python3 extract_frames.py --screen --count 50 --interval 2

    # From a video at specific timestamps:
    python3 extract_frames.py --video match.mp4 --times 10,30,55,120

Output:
    datasets/images/frame_0000.jpg
    datasets/images/frame_0001.jpg
    ...
"""
import cv2
import os
import time
import argparse
import numpy as np

from frame_classifier import classify_frame, FrameType


def is_court_view(frame):
    """
    Backwards-compatible wrapper. Returns True if the frame is classified
    as live court footage (suitable for shuttle detection/training).
    """
    ftype, _ = classify_frame(frame)
    return ftype == FrameType.LIVE_COURT


# --------------------------------------------------------------- video extract
def extract_from_video(video_path, count, output_dir, times=None,
                       court_filter=True, show_rejected=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps
    print(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")
    if court_filter:
        print("Frame classifier: ON (skipping close-ups, replays, hawk-eye, etc.)")
    else:
        print("Frame classifier: OFF")

    if times:
        frame_nums = [int(t * fps) for t in times]
    else:
        margin = int(total_frames * 0.05)
        if count > total_frames - 2 * margin:
            count = total_frames - 2 * margin
        frame_nums = np.linspace(margin, total_frames - margin,
                                 count, dtype=int).tolist()

    os.makedirs(output_dir, exist_ok=True)
    rejected_dir = os.path.join(os.path.dirname(output_dir), "rejected")
    if show_rejected:
        os.makedirs(rejected_dir, exist_ok=True)

    saved = 0
    skipped = 0
    skip_reasons = {}

    # For replay detection we need consecutive frames. Sample 2 frames per
    # timestamp (1 frame apart) so the classifier can compare them.
    prev_gray = None

    for i, fnum in enumerate(frame_nums):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, frame = cap.read()
        if not ret:
            continue

        # Downscale large frames for faster saving + smaller files.
        # 1280px max dimension is plenty for YOLO training.
        max_dim = 1280
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            sc = max_dim / max(h, w)
            frame = cv2.resize(frame, None, fx=sc, fy=sc,
                               interpolation=cv2.INTER_AREA)

        if court_filter:
            ftype, conf = classify_frame(frame, prev_gray=prev_gray)
            if ftype != FrameType.LIVE_COURT:
                skipped += 1
                reason = ftype.value
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                if show_rejected:
                    fname = os.path.join(
                        rejected_dir,
                        f"reject_{skipped:04d}_{reason}.jpg")
                    cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                # Store thumbnail for replay detection next iteration.
                scale = 480.0 / max(frame.shape[:2])
                prev_gray = cv2.resize(
                    frame, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA)
                prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
                continue

        ts = fnum / fps
        fname = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
        cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        # Store thumbnail for replay detection.
        scale = 480.0 / max(frame.shape[:2])
        prev_gray = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)

        if saved % 20 == 0:
            print(f"  saved {saved} (skipped {skipped})  (t={ts:.1f}s)")

    cap.release()
    print(f"\nDone: {saved} court frames saved to {output_dir}/")
    print(f"Filtered: {skipped} frames skipped")
    if skip_reasons:
        print("  Breakdown:")
        for reason, n in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<16} {n}")
    if saved == 0 and skipped > 0:
        print("\nAll frames were filtered. Try --no-filter or --show-rejected")
        print("to see what's being rejected.")


def extract_from_screen(count, interval, output_dir):
    from mss import mss
    from region_selector import select_region

    monitor = select_region(monitor_index=1)
    os.makedirs(output_dir, exist_ok=True)

    sct = mss()
    print(f"\nCapturing {count} frames every {interval}s...")
    print("Press Ctrl+C to stop early.\n")

    for i in range(count):
        shot = sct.grab(monitor)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        fname = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [{i+1}/{count}] saved {fname}", flush=True)
        time.sleep(interval)

    print(f"Done: {count} frames saved to {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Extract frames for labeling")
    ap.add_argument("--video", type=str, help="Video file path")
    ap.add_argument("--screen", action="store_true",
                    help="Capture from screen (livestream)")
    ap.add_argument("--count", type=int, default=100,
                    help="Number of frames to extract")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="Seconds between screen captures")
    ap.add_argument("--times", type=str, default=None,
                    help="Comma-separated timestamps (seconds) for video")
    ap.add_argument("--output", type=str, default="datasets/images",
                    help="Output directory")
    ap.add_argument("--no-filter", action="store_true",
                    help="Disable frame classifier (extract all frames)")
    ap.add_argument("--show-rejected", action="store_true",
                    help="Save rejected frames to datasets/rejected/ for review")
    args = ap.parse_args()

    if args.video:
        times = None
        if args.times:
            times = [float(t) for t in args.times.split(",")]
        extract_from_video(args.video, args.count, args.output, times,
                           court_filter=not args.no_filter,
                           show_rejected=args.show_rejected)
    elif args.screen:
        extract_from_screen(args.count, args.interval, args.output)
    else:
        ap.error("Provide --video or --screen")


if __name__ == "__main__":
    main()
