"""
Cutting the manual work out of labelling, without quietly inventing labels.

Two independent savings, both measured on clip_01 + clip_02 (302 frames) with
the existing hand labels as truth:

1. **Suggestions.** TrackNet is right 97.4% of the time when it fires, so most
   frames do not need a box drawn -- they need a box *confirmed*. Replaying it
   over those clips, 58.3% of frames get a suggestion within 12 px of the hand
   label (3.6 px median) and another 8.6% are correctly empty. That is 66.9% of
   frames reduced to one keypress, leaving 33.1% needing real work.

2. **Sparse labelling.** Shuttle motion between contacts is smooth enough to
   interpolate. Labelling every 3rd frame of clip_01 and filling the rest
   reproduces the held-out labels to 3.8 px median, with 83% landing within
   12 px -- but two tails worth knowing about. About 10% land more than 23 px
   out (worst seen: 150 px), and 8% assert a shuttlecock on a frame the
   original labeller had marked empty, because a straight line between two
   sightings runs straight through the frames where it was not visible.

   That is fine for training data, which tolerates label noise. For an
   evaluation set it is not: those frames would bias the very error metric you
   are trying to measure, and the invented ones would flatter recall. Use
   --stride for training sets, and keep eval sets observed.

Nothing here writes a label on its own. A suggestion is shown, not saved, until
a human accepts it, and interpolated frames are recorded in a sidecar so an
eval set can always be filtered back down to what a person actually looked at.

What does NOT work, both measured and both discarded:

- **Filling the detector's own gaps.** Interpolating across frames TrackNet
  missed puts only ~50% within 12 px (median ~10 px, p90 up to 250). The misses
  are not random: TrackNet fails on fast smashes and motion blur, which is
  exactly where a straight line between neighbours is worst.
- **Anchoring on direction changes.** Forcing extra anchors where the detected
  path bends most cost 6 more points of labelling for no improvement in mean
  error (9.4 -> 9.8 px at stride 3). Curvature measured from detections is
  dominated by detector noise, not by contacts.
"""
import json
import os

# How far apart two anchors may be before a straight line between them stops
# being a fair guess. Past this the frames are left for a human.
DEFAULT_MAX_FILL_GAP = 4

SIDECAR = "_assist.json"


def frame_index(image_path):
    """Trailing integer in a filename, or None: frame_0042.jpg -> 42."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    digits = ""
    for char in reversed(stem):
        if not char.isdigit():
            break
        digits = char + digits
    return int(digits) if digits else None


# Median mean-absolute-difference, on a 32x32 grey thumbnail, above which
# neighbouring frames are not 40 ms apart.
#
# Calibrated on this repo's two regimes: the contiguous clips sit at 0.80-0.91
# and the sampled datasets/images at 4.19-4.84, a ~5x gap that holds at every
# thumbnail size from 32 to 256. 2.0 leaves better than 2x margin either side.
# Individual PAIRS overlap -- a sampled pair can differ by 2.3 and a contiguous
# one by 2.6 -- which is why this is a median over many pairs and not a test
# any single pair can fail.
CONTIGUITY_MAX_DIFF = 2.0


def looks_sequential(image_paths):
    """Whether the filenames are consecutively numbered.

    A necessary condition for contiguous footage, and nowhere near a
    sufficient one: extract_frames.py numbers its output 0000, 0001, 0002 even
    when it sampled those frames minutes apart.
    """
    indices = [frame_index(p) for p in image_paths]
    if any(i is None for i in indices):
        return False
    return all(b - a == 1 for a, b in zip(indices, indices[1:]))


def frame_difference(first_path, second_path):
    """Mean absolute difference between two frames, or None if unreadable."""
    import cv2
    import numpy as np

    first, second = cv2.imread(first_path), cv2.imread(second_path)
    if first is None or second is None:
        return None
    small = lambda image: cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32)).astype(int)
    return float(np.abs(small(first) - small(second)).mean())


def neighbour_difference(image_paths, samples=20):
    """Median difference between neighbouring frames, or None if unreadable."""
    if len(image_paths) < 2:
        return None
    step = max(1, (len(image_paths) - 1) // samples)
    diffs = []
    for i in range(0, len(image_paths) - 1, step):
        diff = frame_difference(image_paths[i], image_paths[i + 1])
        if diff is not None:
            diffs.append(diff)
    if not diffs:
        return None
    diffs.sort()
    return diffs[len(diffs) // 2]


def is_contiguous(image_paths, max_diff=CONTIGUITY_MAX_DIFF, samples=20):
    """Whether these frames are consecutive footage rather than a sample.

    Two things depend on this. TrackNet reads several frames at once -- it
    finds the shuttle partly by where it moved -- so on frames sampled minutes
    apart its suggestions are noise. And contiguous frames must not be split
    between train and val, because neighbours are near-identical.

    The filenames cannot answer it. `datasets/images` is numbered 0000-0999 and
    is not contiguous at all, so this looks at the pixels: consecutive frames
    of a rally barely differ, sampled ones do. The median over several pairs,
    not the mean, so a camera cut inside otherwise-continuous footage does not
    flip the verdict.
    """
    if len(image_paths) < 2 or not looks_sequential(image_paths):
        return False
    median = neighbour_difference(image_paths, samples=samples)
    return median is not None and median <= max_diff


# --------------------------------------------------------------- suggestions
def compute_suggestions(image_paths, box_size, cache_path, model_path=None,
                        progress=True):
    """Run the detector over every frame once, in order, and cache the result.

    Computed up front rather than per frame on demand because the detector is
    temporal: jumping backwards with B would otherwise feed it frames out of
    order and change what it reports for a frame you already saw.

    Returns {basename: (x1, y1, x2, y2) or None}.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as handle:
            cached = json.load(handle)
        if cached.get("count") == len(image_paths):
            return {k: tuple(v) if v else None
                    for k, v in cached["boxes"].items()}

    import cv2

    from tracknet_detector import TrackNetDetector

    detector = TrackNetDetector(model_path) if model_path else TrackNetDetector()
    half = max(box_size, 8) // 2
    boxes = {}
    for number, path in enumerate(image_paths):
        frame = cv2.imread(path)
        if frame is None:
            boxes[os.path.basename(path)] = None
            continue
        found = detector.detect(frame)
        if found is not None and not isinstance(found[0], (int, float)):
            found = found[0]
        if found is None:
            boxes[os.path.basename(path)] = None
        else:
            x, y = int(found[0]), int(found[1])
            boxes[os.path.basename(path)] = (x - half, y - half,
                                             x + half, y + half)
        if progress and number % 25 == 0:
            print(f"  suggesting… {number}/{len(image_paths)}",
                  end="\r", flush=True)

    if progress:
        found_count = sum(1 for b in boxes.values() if b)
        print(f"  suggestions ready: {found_count}/{len(image_paths)} frames "
              f"have one" + " " * 20)
    if cache_path:
        with open(cache_path, "w") as handle:
            json.dump({"count": len(image_paths), "boxes": boxes}, handle)
    return boxes


# ------------------------------------------------------------ sparse filling
def anchor_indices(image_paths, stride):
    """Positions a human is asked to label when working at `stride`.

    The first and last frame are always included: interpolation needs a labelled
    frame on both sides, and anything outside the anchored span cannot be filled
    at all.
    """
    if stride <= 1:
        return set(range(len(image_paths)))
    anchors = set(range(0, len(image_paths), stride))
    anchors.add(0)
    anchors.add(len(image_paths) - 1)
    return anchors


def interpolate(points, max_gap=DEFAULT_MAX_FILL_GAP):
    """Fill gaps between labelled positions.

    `points` maps position -> (x, y) for frames that have a shuttlecock, and
    position -> None for frames confirmed to have none. Positions absent from
    the mapping are unlabelled and are what this fills.

    A gap is filled only when both neighbours hold a position and they are no
    more than `max_gap` apart. Two labels far apart say nothing reliable about
    what happened between them, and a confirmed-empty neighbour means the
    shuttle was not visible, so there is nothing to interpolate towards.

    Returns {position: (x, y)} for the frames it is willing to fill.
    """
    known = sorted(k for k, v in points.items() if v is not None)
    filled = {}
    for low, high in zip(known, known[1:]):
        if high - low <= 1 or high - low > max_gap:
            continue
        # Only fill frames nobody has looked at: never overwrite a human's
        # "no shuttle here", which is a real observation.
        if any(points.get(k) is None and k in points
               for k in range(low + 1, high)):
            continue
        for position in range(low + 1, high):
            if position in points:
                continue
            ratio = (position - low) / (high - low)
            filled[position] = (
                points[low][0] + ratio * (points[high][0] - points[low][0]),
                points[low][1] + ratio * (points[high][1] - points[low][1]),
            )
    return filled


# ------------------------------------------------------------------ sidecar
def load_sidecar(labels_dir):
    """Which labels were accepted from a suggestion, and which were filled in."""
    path = os.path.join(labels_dir, SIDECAR)
    if not os.path.exists(path):
        return {"accepted": [], "filled": []}
    with open(path) as handle:
        data = json.load(handle)
    data.setdefault("accepted", [])
    data.setdefault("filled", [])
    return data


def save_sidecar(labels_dir, data):
    """Record provenance next to the labels.

    An evaluation set has to be able to say which of its labels a person
    actually looked at. Without this the interpolated frames are
    indistinguishable from observed ones, and any error metric computed over
    them is partly measuring the interpolation.
    """
    path = os.path.join(labels_dir, SIDECAR)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=1)
