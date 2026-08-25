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


def build_dataset_yaml(images_dir, labels_dir, output_path="datasets/data.yaml"):
    """
    Build the data.yaml file YOLO expects, and reorganize data into
    train/val splits.
    """
    images_dir = os.path.abspath(images_dir)
    labels_dir = os.path.abspath(labels_dir)

    # Collect all image filenames that have corresponding labels
    all_images = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png"))
    )
    random.shuffle(all_images)

    # 80/20 train/val split
    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    # Create YOLO directory structure
    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out = os.path.join("datasets", "yolo", split, "images")
        lbl_out = os.path.join("datasets", "yolo", split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_name in imgs:
            # Symlink or copy image
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(img_out, img_name)
            if os.path.exists(dst_img):
                continue
            try:
                os.symlink(src_img, dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

            # Copy corresponding label
            base = os.path.splitext(img_name)[0]
            src_lbl = os.path.join(labels_dir, base + ".txt")
            dst_lbl = os.path.join(lbl_out, base + ".txt")
            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                try:
                    os.symlink(src_lbl, dst_lbl)
                except OSError:
                    shutil.copy2(src_lbl, dst_lbl)

    yolo_root = os.path.abspath("datasets/yolo")
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
    ap.add_argument("--device", type=str, default=None,
                    help="Device: cpu, mps, cuda, or auto")
    args = ap.parse_args()

    check_prerequisites(args.base, args.images, args.labels)
    data_yaml = build_dataset_yaml(args.images, args.labels)

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
