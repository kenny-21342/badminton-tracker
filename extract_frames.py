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


# --------------------------------------------------------------- rally finding
COURT_SIZE = (320, 180)


def _small_gray(frame):
    return cv2.cvtColor(cv2.resize(frame, COURT_SIZE),
                        cv2.COLOR_BGR2GRAY).astype(np.int16)


def build_court_reference(video_path, samples=240, cluster_tol=12.0):
    """
    Build a picture of the empty wide broadcast court.

    A plain median over sampled frames does not work: the sample spans crowd
    shots, close-ups and replays, and the median of all of them is a grey
    smear that sits about equally far from everything. Instead take the
    largest cluster of mutually similar frames -- the wide court view is the
    one shot the director keeps returning to -- and median only those. The
    players move around within it, so they average out and the court remains.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open '{video_path}'")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for f in np.linspace(total * 0.03, total * 0.97, samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if ok:
            frames.append(_small_gray(fr))
    cap.release()

    stack = np.stack(frames)
    dist = np.abs(stack[:, None] - stack[None, :]).mean(axis=(2, 3))
    neighbours = (dist < cluster_tol).sum(1)
    seed = int(neighbours.argmax())
    members = stack[dist[seed] < cluster_tol]
    print(f"Court reference: modal shot is {len(members)}/{len(stack)} "
          f"sampled frames")
    return np.median(members, axis=0).astype(np.int16)


def court_scores(video_path, reference, stride=2, progress=True):
    """Mean absolute difference from the empty court, for every strided frame."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores = {}
    fnum = 0
    while True:
        if not cap.grab():
            break
        if fnum % stride == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            scores[fnum] = float(
                np.abs(_small_gray(fr) - reference).mean())
            if progress and len(scores) % 5000 == 0:
                print(f"  {fnum}/{total} frames scored")
        fnum += 1
    cap.release()
    return scores


def _otsu(values):
    """Split a bimodal score distribution without hand-picking a number."""
    v = np.clip(np.asarray(values), 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(v.reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)


def find_rallies(video_path, stride=2, min_seconds=4.0, max_clips=10,
                 clip_seconds=6.0, threshold=None, margin_frames=6):
    """
    Find contiguous stretches of the wide court view.

    This does not use classify_frame. That test asks whether a court is
    visible, which is true of a coaching break at the net and of the tight
    low-angle shot the director cuts to on a smash -- so it accepts both, and
    a clip built from it silently spans camera cuts. Distance from the empty
    wide court separates them outright: on the 2011 final a clean rally scores
    8-12 and an interval at the net scores 57, with nothing in between.

    A cut needs no special handling: cutting away raises the score, so a run
    that stays below threshold is by construction a single unbroken shot.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    reference = build_court_reference(video_path)
    print(f"Scoring every {stride}%s frame against it..."
          % ("nd" if stride == 2 else "th"))
    scores = court_scores(video_path, reference, stride=stride)

    values = np.array(list(scores.values()))
    if threshold is not None:
        thr, how = threshold, "given"
    else:
        # Otsu splits wide-court from everything else, but it splits the whole
        # distribution and lands well above where clean rallies actually sit,
        # so it lets in the near-court shots the director cuts to. Treat it as
        # a starting point and check the per-clip worst score.
        thr, how = _otsu(values), "Otsu"
    on_court = (values < thr).mean()
    print(f"Threshold {thr:.1f} ({how}); {on_court * 100:.0f}% of frames are "
          f"wide court view")

    keys = sorted(scores)
    runs = []
    run_start = None
    for k in keys:
        if scores[k] < thr:
            if run_start is None:
                run_start = k
        else:
            if run_start is not None:
                runs.append((run_start, k - stride))
                run_start = None
    if run_start is not None:
        runs.append((run_start, keys[-1]))

    # Pull in from both ends: the scan is strided, so a boundary is only
    # located to within `stride` frames, and the frames either side of a cut
    # are the blurred ones.
    runs = [(a + margin_frames, b - margin_frames) for a, b in runs]
    min_frames = int(min_seconds * fps)
    long_runs = [r for r in runs if r[1] - r[0] >= min_frames]
    long_runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    print(f"{len(runs)} unbroken wide-court shots, {len(long_runs)} at least "
          f"{min_seconds}s long")

    # Spread the picks across the match rather than taking the longest few,
    # which tend to cluster in one game.
    chosen = []
    for r in long_runs:
        mid = (r[0] + r[1]) / 2
        if all(abs(mid - (c[0] + c[1]) / 2) > 60 * fps for c in chosen):
            chosen.append(r)
        if len(chosen) >= max_clips:
            break
    chosen.sort()

    half = clip_seconds * fps / 2
    out = []
    for a, b in chosen:
        mid = (a + b) / 2
        out.append((max(a, mid - half) / fps, min(b, mid + half) / fps))
    return out, reference


# ---------------------------------------------------------------- clip extract
def extract_clips(video_path, ranges, output_dir, reference=None):
    """
    Write every frame in each (start, end) second range to its own directory.

    Reads sequentially -- seeking per frame is both slow and unreliable on
    long H.264 files. Each clip lands in its own subdirectory so temporal
    order is unambiguous and clip boundaries survive on disk, which is what
    evaluate.py needs to reset a tracker between clips instead of bridging
    a cut.

    With a reference, every written frame is scored against the empty court
    and any clip that leaves the wide view is reported. Verify this rather
    than trusting it: a clip that spans a cut makes every detector look
    broken for a reason that is not the detector's fault.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open '{video_path}'")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bounds = [(int(s * fps), int(e * fps)) for s, e in ranges]
    last_frame = max(b for _, b in bounds)

    os.makedirs(output_dir, exist_ok=True)
    counts = [0] * len(bounds)
    worst = [0.0] * len(bounds)
    fnum = 0
    written = []

    while fnum <= last_frame:
        if not cap.grab():
            break
        active = [i for i, (a, b) in enumerate(bounds) if a <= fnum <= b]
        if active:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if reference is not None:
                s = float(np.abs(_small_gray(frame) - reference).mean())
                for i in active:
                    worst[i] = max(worst[i], s)
            max_dim = 1280
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                sc = max_dim / max(h, w)
                frame = cv2.resize(frame, None, fx=sc, fy=sc,
                                   interpolation=cv2.INTER_AREA)
            for i in active:
                d = os.path.join(output_dir, f"clip_{i + 1:02d}")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"frame_{counts[i]:04d}.jpg"),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                counts[i] += 1
        fnum += 1
    cap.release()

    print()
    for i, (a, b) in enumerate(bounds):
        d = os.path.join(output_dir, f"clip_{i + 1:02d}")
        note = ""
        if reference is not None:
            note = f"  worst court-diff {worst[i]:5.1f}"
            if worst[i] > 25:
                note += "  <-- LEAVES THE WIDE VIEW, do not label this clip"
        print(f"clip_{i + 1:02d}  {a / fps:7.1f}s - {b / fps:7.1f}s  "
              f"{counts[i]:4d} frames{note}")
        written.append(d)
    print(f"\nTotal {sum(counts)} frames in {len(bounds)} clips -> {output_dir}")
    return written


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
    ap.add_argument("--find-rallies", type=int, metavar="N",
                    help="Scan the video for N contiguous live-court stretches "
                         "and print them as a --clips string")
    ap.add_argument("--clips", type=str, default=None,
                    help="Contiguous ranges in seconds, e.g. '120-126,300-306'. "
                         "Every frame in range is written, one dir per clip.")
    ap.add_argument("--clip-seconds", type=float, default=6.0,
                    help="Length of each clip produced by --find-rallies")
    ap.add_argument("--stride", type=int, default=2,
                    help="Frame stride when scanning for rallies")
    ap.add_argument("--court-threshold", type=float, default=None,
                    help="Override the automatic wide-court cutoff")
    args = ap.parse_args()

    if args.video and args.find_rallies:
        ranges, reference = find_rallies(args.video, stride=args.stride,
                                         max_clips=args.find_rallies,
                                         clip_seconds=args.clip_seconds,
                                         threshold=args.court_threshold)
        if not ranges:
            print("No unbroken rally found. Try --clip-seconds 4.")
            return
        spec = ",".join(f"{s:.1f}-{e:.1f}" for s, e in ranges)
        print(f'\n--clips "{spec}"')
        if args.output != "datasets/images":
            extract_clips(args.video, ranges, args.output, reference)
        return

    if args.video and args.clips:
        ranges = []
        for part in args.clips.split(","):
            a, b = part.split("-")
            ranges.append((float(a), float(b)))
        extract_clips(args.video, ranges, args.output,
                      build_court_reference(args.video))
        return

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
