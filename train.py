"""
Train a YOLOv8-nano model to detect the shuttlecock.

This fine-tunes the pre-trained YOLOv8n on your labeled frames, producing
a model at runs/detect/train/weights/best.pt that main_screen.py loads
automatically with the --yolo flag.

Prerequisites:
  1. yolov8n.pt in the project directory (base model for fine-tuning)
     Download from: https://github.com/ultralytics/assets/releases/
                    download/v8.4.0/yolov8n.pt
  2. Labeled frames in datasets/images/ + datasets/labels/
     (use extract_frames.py then labeler.py)

Usage:
    python3 train.py                        # full pipeline
    python3 train.py --add clip_03:labels_03   # fold in extra clips
    python3 train.py --epochs 100           # more epochs
    python3 train.py --base yolov8s.pt      # bigger base model (more accurate)
    python3 train.py --imgsz 480            # smaller images (faster training)

After training:
    python3 main_screen.py --yolo           # run with your trained model
"""
import os
import sys
import shutil
import argparse
import random

import label_assist


def check_prerequisites(base_model, images_dir, labels_dir):
    missing = []

    if not os.path.exists(base_model):
        missing.append(
            f"  {base_model} — download from:\n"
            f"    https://github.com/ultralytics/assets/releases/"
            f"download/v8.4.0/{base_model}"
        )

    images = [f for ext in ("*.jpg", "*.png")
              for f in os.path.splitext(ext)]
    img_files = []
    for ext in ("*.jpg", "*.png"):
        img_files.extend(os.listdir(images_dir)
                         if os.path.isdir(images_dir) else [])
    img_files = []
    if os.path.isdir(images_dir):
        for f in os.listdir(images_dir):
            if f.lower().endswith((".jpg", ".png")):
                img_files.append(f)

    if not img_files:
        missing.append(
            f"  No images in {images_dir}/ — run extract_frames.py first")

    # Check at least some labels exist
    label_count = 0
    if os.path.isdir(labels_dir):
        label_count = sum(
            1 for f in os.listdir(labels_dir) if f.endswith(".txt"))

    if label_count == 0:
        missing.append(
            f"  No labels in {labels_dir}/ — run labeler.py first")

    if missing:
        print("Missing prerequisites:\n")
        print("\n".join(missing))
        print("\nWorkflow:")
        print("  1. Download base model (see URL above)")
        print("  2. python3 extract_frames.py --video match.mp4 --count 200")
        print("  3. python3 labeler.py")
        print("  4. python3 train.py")
        sys.exit(1)

    # Count how many have actual annotations (non-empty)
    annotated = 0
    for f in os.listdir(labels_dir):
        if f.endswith(".txt"):
            path = os.path.join(labels_dir, f)
            if os.path.getsize(path) > 0:
                annotated += 1

    print(f"Dataset: {len(img_files)} images, "
          f"{annotated} annotated, {label_count - annotated} empty")
    if annotated < 30:
        print(f"\nWarning: only {annotated} labeled images. "
              f"Label at least 50-100 for good results.")

    return img_files


# Clips evaluate.py measures against. Training on them makes every number it
# reports meaningless, and the mistake is invisible afterwards.
EVAL_CLIPS = ("clip_01", "clip_02")


def warn_if_evaluation_data(images_dir):
    """Say so, loudly, if a source is part of the evaluation set."""
    name = os.path.basename(os.path.normpath(images_dir))
    if name in EVAL_CLIPS and "eval_" in images_dir:
        print(f"  *** {name} is part of the evaluation set that "
              f"evaluate.py --clips measures against.")
        print(f"  *** Training on it will inflate every number that reports, "
              f"and nothing downstream will notice.")
        return True
    return False


def collect_sources(pairs):
    """Read every (images, labels) pair into split groups.

    A group is a set of frames that must not be divided between train and
    val. Contiguous footage is one group per directory: neighbouring frames of
    a rally are near-identical, so splitting them at random puts a frame in
    train and the frame 40 ms later in val, and the validation score stops
    measuring generalisation. Sampled frames are far enough apart to split
    individually, so each is its own group.
    """
    groups = []
    for images_dir, labels_dir in pairs:
        images_dir = os.path.abspath(images_dir)
        labels_dir = os.path.abspath(labels_dir)
        names = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".png"))
            and os.path.exists(
                os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt"))
        )
        if not names:
            print(f"  warning: no labelled frames in {images_dir}")
            continue

        warn_if_evaluation_data(images_dir)
        paths = [os.path.join(images_dir, n) for n in names]
        contiguous = label_assist.is_contiguous(paths)
        # Print the measurement, not just the verdict: the threshold is a
        # calibrated heuristic, and a mislabelled source either leaks frames
        # across the split or wastes the split entirely.
        spread = label_assist.neighbour_difference(paths)
        detail = f"neighbour diff {spread:.2f}" if spread is not None else "unread"
        if contiguous:
            groups.append((os.path.basename(images_dir),
                           images_dir, labels_dir, names))
            print(f"  {os.path.basename(images_dir)}: {len(names)} frames "
                  f"({detail} — contiguous, kept together)")
        else:
            for n in names:
                groups.append((n, images_dir, labels_dir, [n]))
            print(f"  {os.path.basename(images_dir)}: {len(names)} frames "
                  f"({detail} — sampled, split individually)")
    return groups


def build_dataset_yaml(pairs, output_path="datasets/data.yaml", val_fraction=0.2,
                       seed=0):
    """Build data.yaml and lay out train/val, splitting by group.

    The output directory is rebuilt from scratch every time. It used to be
    written incrementally, skipping any file already present, which meant a
    frame that landed in val on one run and train on the next was left in
    both — silently training on part of the validation set.
    """
    print("Collecting sources:")
    groups = collect_sources(pairs)
    if not groups:
        sys.exit("No labelled frames found.")

    random.Random(seed).shuffle(groups)
    total = sum(len(g[3]) for g in groups)
    target_val = total * val_fraction

    val_groups, train_groups, val_count = [], [], 0
    for group in groups:
        # Whole groups only: a contiguous clip goes entirely to one side.
        if val_count < target_val:
            val_groups.append(group)
            val_count += len(group[3])
        else:
            train_groups.append(group)

    yolo_root = os.path.abspath("datasets/yolo")
    if os.path.exists(yolo_root):
        shutil.rmtree(yolo_root)

    counts = {}
    for split, split_groups in [("train", train_groups), ("val", val_groups)]:
        img_out = os.path.join(yolo_root, split, "images")
        lbl_out = os.path.join(yolo_root, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        written = 0
        for group_name, images_dir, labels_dir, names in split_groups:
            for img_name in names:
                stem = os.path.splitext(img_name)[0]
                # Frames from different clips can share a filename, so the
                # group name is part of the destination.
                unique = f"{group_name}__{img_name}" if group_name != img_name \
                    else img_name
                unique_stem = os.path.splitext(unique)[0]
                for src, dst in (
                    (os.path.join(images_dir, img_name),
                     os.path.join(img_out, unique)),
                    (os.path.join(labels_dir, stem + ".txt"),
                     os.path.join(lbl_out, unique_stem + ".txt")),
                ):
                    if not os.path.exists(src):
                        continue
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                written += 1
        counts[split] = written

    train_path = os.path.join(yolo_root, "train", "images")
    val_path = os.path.join(yolo_root, "val", "images")

    yaml_content = (
        f"path: {yolo_root}\n"
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        f"\n"
        f"names:\n"
        f"  0: shuttlecock\n"
    )

    with open(output_path, "w") as f:
        f.write(yaml_content)

    train_imgs = [None] * counts["train"]
    val_imgs = [None] * counts["val"]
    print(f"  {len(train_groups)} train groups / {len(val_groups)} val groups")
    print(f"Dataset split: {len(train_imgs)} train / {len(val_imgs)} val")
    print(f"data.yaml written to {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8 shuttlecock detector")
    ap.add_argument("--base", type=str, default="yolov8n.pt",
                    help="Base model (yolov8n.pt recommended for real-time)")
    ap.add_argument("--epochs", type=int, default=80,
                    help="Training epochs (50-150 typical)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Training image size")
    ap.add_argument("--batch", type=int, default=16,
                    help="Batch size (lower if out of memory)")
    ap.add_argument("--images", type=str, default="datasets/images",
                    help="Images directory")
    ap.add_argument("--labels", type=str, default="datasets/labels",
                    help="Labels directory")
    ap.add_argument("--add", action="append", default=[], metavar="IMAGES:LABELS",
                    help="Extra source, e.g. "
                         "datasets/eval_ld2011/clip_03:datasets/eval_ld2011/"
                         "labels_03. Repeatable. Contiguous clips are kept "
                         "whole when splitting train/val.")
    ap.add_argument("--val-fraction", type=float, default=0.2,
                    help="Share of frames held out for validation.")
    ap.add_argument("--device", type=str, default=None,
                    help="Device: cpu, mps, cuda, or auto")
    args = ap.parse_args()

    check_prerequisites(args.base, args.images, args.labels)

    pairs = [(args.images, args.labels)]
    for extra in args.add:
        if ":" not in extra:
            sys.exit(f"--add expects IMAGES:LABELS, got {extra!r}")
        images, labels = extra.rsplit(":", 1)
        pairs.append((images, labels))

    data_yaml = build_dataset_yaml(pairs, val_fraction=args.val_fraction)

    from ultralytics import YOLO

    # Auto-detect MPS on Apple Silicon.
    device = args.device
    if device is None:
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
                print(f"Using MPS (Apple Silicon GPU) for training")
        except Exception:
            pass
    if device is None:
        device = "cpu"
        print(f"WARNING: Using CPU for training (slow). MPS not available.")

    model = YOLO(args.base)
    print(f"\nTraining {args.base} for {args.epochs} epochs on {device}...")
    print(f"Results will be saved to runs/detect/train/")

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project="runs/detect",
        name="train",
        exist_ok=True,
        # Augmentations tuned for small fast objects
        hsv_h=0.015,    # slight hue shift (stadium lighting varies)
        hsv_s=0.5,      # saturation shift
        hsv_v=0.3,      # brightness shift
        degrees=2.0,    # slight rotation
        translate=0.1,
        scale=0.3,
        fliplr=0.5,     # horizontal flip (court is roughly symmetric)
        flipud=0.0,     # no vertical flip (gravity matters!)
        mosaic=0.5,
        mixup=0.0,
        # Small object tuning
        close_mosaic=10,  # disable mosaic last 10 epochs for fine-tuning
    )

    best_path = "runs/detect/train/weights/best.pt"
    print(f"\nTraining complete!")
    print(f"Best model: {best_path}")
    print(f"\nTest it with:")
    print(f"  python3 main_screen.py --yolo")


if __name__ == "__main__":
    main()
