#!/usr/bin/env python3
"""
Measure how well each detector actually finds the shuttlecock.

Until now the three detection modes were compared by watching the overlay and
forming an impression. This runs them over labelled frames and reports numbers,
so "pick a default" is a decision with evidence behind it.

    python3 evaluate.py                      # all detectors, sampled set
    python3 evaluate.py --clips              # the contiguous clip set
    python3 evaluate.py --detector tracknet  # just one
    python3 evaluate.py --limit 200          # quick pass

Two things about the method are worth stating, because they decide whether the
numbers mean anything.

**Frames are fed in order.** Classic CV needs motion between consecutive
frames and TrackNet reads three at a time; both collapse on a shuffled set. A
random train/val split — which is what datasets/yolo holds — would therefore
flatter YOLO and punish the other two for no reason. So the evaluation runs
over datasets/images in filename order.

**YOLO's numbers on that set are contaminated.** It was fine-tuned on 800 of
these 1000 frames, so it has seen most of the answers. Its row is reported
anyway, marked, and also re-reported on the 200-frame validation split it did
not train on. TrackNet was pre-trained on unrelated broadcast footage and
classic CV learns nothing, so for those two the set is honest held-out data.

**`--clips` is the layout that actually tests a tracker.** datasets/images is
1000 frames sampled evenly across the match, so its median run of consecutive
footage is one frame — a set on which a temporal detector can barely be
measured at all. datasets/eval_ld2011 holds 151-frame contiguous clips
instead, one run each. Only the clips with a populated labels_NN directory are
used; the rest are named and skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "datasets" / "images"
LABELS = ROOT / "datasets" / "labels"
# Contiguous clips: datasets/eval_ld2011/clip_NN holds 151 consecutive frames
# and labels_NN holds their labels. This is the layout the numbers should
# eventually come from — see load_clips.
EVAL_CLIPS = ROOT / "datasets" / "eval_ld2011"

# Distances, in pixels of the 1280x720 source, at which a detection counts as
# correct. The shuttlecock's labelled box is around 16x16, so 8 px is "on it",
# 25 px is "close enough to draw a usable trail", and 50 px is "the right
# general place". Reporting all three avoids a single arbitrary threshold
# deciding which detector looks best.
TOLERANCES = (8, 25, 50)
PRIMARY_TOLERANCE = 25


# Mean absolute pixel difference (on a quarter-scale grey frame) above which
# two consecutive files are not consecutive *footage*. Adjacent video frames
# sit around 2-5; a cut to another rally sits above 25.
CUT_THRESHOLD = 12.0
# A temporal detector needs a few frames of context before it can be judged.
MIN_RUN = 4


@dataclass
class Frame:
    path: Path
    truth: tuple[float, float] | None  # None when no shuttlecock is visible
    starts_run: bool = False           # first frame after a cut
    clip: str = ""                     # clip_NN in the contiguous set, else ""


@dataclass
class Result:
    name: str
    frames: int = 0
    # Frames where a shuttlecock is present.
    positives: int = 0
    # Frames where none is present, which is where false positives come from.
    negatives: int = 0
    hits: dict[int, int] = field(default_factory=lambda: {t: 0 for t in TOLERANCES})
    detections_on_positive: int = 0
    detections_on_negative: int = 0
    errors: list[float] = field(default_factory=list)
    seconds: float = 0.0
    failed: str | None = None

    def recall(self, tolerance: int) -> float:
        return self.hits[tolerance] / self.positives if self.positives else 0.0

    def precision(self, tolerance: int) -> float:
        """Of everything it pointed at, how much was right.

        A detection on a frame with no shuttlecock is always wrong; one on a
        frame that has one is wrong if it is further away than the tolerance.
        """
        claimed = self.detections_on_positive + self.detections_on_negative
        return self.hits[tolerance] / claimed if claimed else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.detections_on_negative / self.negatives if self.negatives else 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


def load_frames(images: Path, labels: Path, limit: int | None) -> list[Frame]:
    """Read the labelled set in filename order.

    A zero-byte label file means the frame was reviewed and has no shuttlecock
    in it — a negative, not a gap. 270 of the 1000 are like that, and they are
    the only thing measuring false positives.
    """
    frames: list[Frame] = []
    for image_path in sorted(images.glob("*.jpg")):
        label_path = labels / f"{image_path.stem}.txt"
        truth = None
        if label_path.exists() and label_path.stat().st_size > 0:
            first = label_path.read_text().strip().splitlines()[0].split()
            if len(first) >= 3:
                truth = (float(first[1]), float(first[2]))  # normalised cx, cy
        frames.append(Frame(image_path, truth))
        if limit and len(frames) >= limit:
            break

    mark_runs(frames)
    return frames


def load_clips(root: Path, limit: int | None) -> tuple[list[Frame], list[str]]:
    """Read the contiguous-clip set, one run per clip.

    Every clip_NN is 151 consecutive frames of footage, so unlike
    datasets/images there is nothing to guess about where the runs are: each
    clip is one, and the boundary between two clips is always a cut. Cut
    detection still runs *inside* a clip, because a broadcast replay can start
    mid-clip, but it never compares the last frame of one clip against the
    first of the next.

    A clip is only usable if labels_NN exists and holds at least one label
    file. A directory of zero .txt files is unlabelled footage, not 151
    frames that were reviewed and found empty, and counting it as negatives
    would invent a perfect false-positive score out of nothing. Those clips
    are skipped and named in the second return value.
    """
    frames: list[Frame] = []
    skipped: list[str] = []
    for clip_dir in sorted(root.glob("clip_*")):
        labels_dir = root / clip_dir.name.replace("clip_", "labels_")
        if not labels_dir.is_dir() or not any(labels_dir.glob("*.txt")):
            skipped.append(clip_dir.name)
            continue

        clip_frames: list[Frame] = []
        for image_path in sorted(clip_dir.glob("*.jpg")):
            label_path = labels_dir / f"{image_path.stem}.txt"
            truth = None
            if label_path.exists() and label_path.stat().st_size > 0:
                first = label_path.read_text().strip().splitlines()[0].split()
                if len(first) >= 3:
                    truth = (float(first[1]), float(first[2]))
            clip_frames.append(Frame(image_path, truth, clip=clip_dir.name))

        mark_runs(clip_frames)
        if clip_frames:
            clip_frames[0].starts_run = True  # a clip boundary is always a cut
        frames.extend(clip_frames)
        if limit and len(frames) >= limit:
            del frames[limit:]
            break

    return frames, skipped


def mark_runs(frames: list[Frame]) -> None:
    """Flag where the footage jumps, so detectors can be reset there.

    datasets/images is sampled frames, not clips: consecutive files are
    sometimes adjacent video frames and sometimes a cut to a different rally.
    Classic CV compares against the previous frame and TrackNet reads three at
    a time, so running them straight through treats every cut as a violent
    movement of the shuttlecock. Detecting the cuts lets each run be measured
    on its own.
    """
    previous = None
    for frame in frames:
        image = cv2.imread(str(frame.path))
        if image is None:
            continue
        small = cv2.cvtColor(
            cv2.resize(image, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        if previous is None or float(np.abs(small - previous).mean()) > CUT_THRESHOLD:
            frame.starts_run = True
        previous = small


def build(name: str):
    """Construct one detector, or raise with a readable reason."""
    if name == "cv":
        from detector import ShuttlecockDetector

        return ShuttlecockDetector()
    if name == "tracknet":
        from tracknet_detector import TrackNetDetector

        weights = ROOT / "models" / "TrackNet_best.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"{weights} is missing — run python3 download_tracknet.py first"
            )
        return TrackNetDetector(str(weights))
    if name == "yolo":
        from yolo_detector import YOLODetector

        return YOLODetector()
    raise ValueError(f"unknown detector {name!r}")


def output_scale(detector) -> float:
    """Factor converting a detector's output back to full-frame pixels.

    The classic detector works entirely in downscaled space and returns
    coordinates there — main_screen.py divides by the same factor before
    drawing (see its comment at the detect call). TrackNet and YOLO already
    map back to full frame themselves. Missing this made classic CV look like
    it was 539 px out on every frame.
    """
    scale = getattr(detector, "scale", None)
    if scale is None:
        classic = getattr(detector, "classic", None)
        scale = getattr(classic, "scale", None)
    return 1.0 / scale if scale else 1.0


def evaluate(name: str, frames: list[Frame]) -> Result:
    result = Result(name=name)
    try:
        detector = build(name)
    except Exception as error:
        result.failed = f"{type(error).__name__}: {error}"
        return result
    to_full_frame = output_scale(detector)

    # YOLO reads one frame at a time and needs no run-up; the other two do.
    # Applying the warm-up filter to YOLO would throw away most of a set whose
    # median run is a single frame, for nothing.
    temporal = name in {"cv", "tracknet"}

    started = time.time()
    since_cut = 0
    for frame in frames:
        image = cv2.imread(str(frame.path))
        if image is None:
            continue
        height, width = image.shape[:2]

        if frame.starts_run:
            # A cut is not motion. Clear the detector's history rather than let
            # it chase the difference between two unrelated rallies. reset()
            # rather than rebuild: reconstructing TrackNet reloads 130 MB of
            # weights, which took the harness from ~40 fps to 1.9.
            if hasattr(detector, "reset"):
                detector.reset()
            else:
                detector = build(name)
                to_full_frame = output_scale(detector)
            since_cut = 0
        since_cut += 1
        result.frames += 1

        try:
            point = detector.detect(image)
        except Exception as error:  # one bad frame should not void the run
            result.failed = f"{type(error).__name__}: {error}"
            break

        if point is not None:
            point = (point[0] * to_full_frame, point[1] * to_full_frame)

        # The first frames after a cut are context, not a fair test.
        if temporal and since_cut < MIN_RUN:
            result.frames -= 1
            continue

        if frame.truth is None:
            result.negatives += 1
            if point is not None:
                result.detections_on_negative += 1
            continue

        result.positives += 1
        truth = (frame.truth[0] * width, frame.truth[1] * height)
        if point is None:
            continue
        result.detections_on_positive += 1
        distance = float(np.hypot(point[0] - truth[0], point[1] - truth[1]))
        result.errors.append(distance)
        for tolerance in TOLERANCES:
            if distance <= tolerance:
                result.hits[tolerance] += 1

    result.seconds = time.time() - started
    return result


def report(results: list[Result], contaminated: set[str], note: str) -> None:
    print()
    print(f"{'detector':10} {'recall':>8} {'prec':>7} {'FP rate':>8} "
          f"{'median err':>11} {'PCK@8':>7} {'PCK@25':>7} {'PCK@50':>7} {'fps':>6}")
    print("-" * 82)
    for result in results:
        if result.failed and result.frames == 0:
            print(f"{result.name:10} did not run — {result.failed[:60]}")
            continue
        median = float(np.median(result.errors)) if result.errors else float("nan")
        mark = " *" if result.name in contaminated else ""
        print(
            f"{result.name:10} "
            f"{result.recall(PRIMARY_TOLERANCE):8.1%} "
            f"{result.precision(PRIMARY_TOLERANCE):7.1%} "
            f"{result.false_positive_rate:8.1%} "
            f"{median:10.1f}px "
            f"{result.recall(8):7.1%} "
            f"{result.recall(25):7.1%} "
            f"{result.recall(50):7.1%} "
            f"{result.fps:6.1f}{mark}"
        )
        if result.failed:
            print(f"{'':10} stopped early — {result.failed[:60]}")
    print()
    print(f"recall/prec/PCK at {PRIMARY_TOLERANCE}px unless stated. "
          f"FP rate is detections on frames with no shuttlecock.")
    if contaminated:
        print(note)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector", action="append",
        choices=["cv", "tracknet", "yolo"],
        help="repeatable; defaults to all three",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N frames, for a quick pass")
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument(
        "--clips", type=Path, nargs="?", const=EVAL_CLIPS, default=None,
        metavar="DIR",
        help="evaluate the contiguous clip_NN/labels_NN set instead "
             f"(default {EVAL_CLIPS.name})",
    )
    args = parser.parse_args()

    skipped: list[str] = []
    if args.clips is not None:
        if not args.clips.is_dir():
            print(f"no clip set at {args.clips}", file=sys.stderr)
            return 1
        frames, skipped = load_clips(args.clips, args.limit)
        if not frames:
            print(f"no labelled clips under {args.clips}", file=sys.stderr)
            return 1
    else:
        if not args.images.is_dir():
            print(f"no labelled frames at {args.images}", file=sys.stderr)
            return 1
        frames = load_frames(args.images, args.labels, args.limit)

    positives = sum(1 for frame in frames if frame.truth)
    clips = sorted({frame.clip for frame in frames if frame.clip})
    where = f" over {len(clips)} clips ({', '.join(clips)})" if clips else ""
    print(f"{len(frames)} frames{where} — {positives} with a shuttlecock, "
          f"{len(frames) - positives} without")
    if skipped:
        print(f"skipped {len(skipped)} unlabelled clip(s): {', '.join(skipped)}")

    names = args.detector or ["cv", "tracknet", "yolo"]
    results = []
    for name in names:
        print(f"  running {name}…", flush=True)
        results.append(evaluate(name, frames))

    # YOLO was fine-tuned on 800 of the 1000 frames in the default set, so it
    # has literally seen those answers. The clip set holds none of the same
    # files, but it is cut from the same match, so frames near a training
    # sample are near-duplicates of it — weaker contamination, still not a
    # clean held-out measurement. Mark both, differently.
    if args.clips is not None:
        contaminated = {"yolo"}
        note = ("* fine-tuned on frames sampled from this same match. No file "
                "here was in its training set, but near-duplicates were.")
    else:
        contaminated = {"yolo"} if args.images == IMAGES else set()
        note = ("* trained on most of these frames — treat as an upper bound, "
                "not a measurement.")
    report(results, contaminated, note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
