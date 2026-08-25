"""
TrackNet Pre-trained Weights Downloader.

Downloads TrackNetV2 PyTorch weights trained on professional badminton
broadcast footage, from the open TrackNet Series PyTorch project:

    https://github.com/AnInsomniacy/tracknet-series-pytorch

The weights are saved to models/TrackNet_best.pt, which main_screen.py loads
when run with --tracknet. The downloader resumes partial downloads and
validates that the file is real trained weights (not random initialization).
"""

import os
import subprocess
import sys

import torch

# Direct download links for pre-trained TrackNetV2 weights.
# Architecture: 3-frame (9-channel) input, 3 sigmoid heatmap outputs,
# 512x288 native resolution - identical to the TrackNet class used here.
WEIGHT_SOURCES = [
    {
        "name": "TrackNetV2 pre-trained (badminton)",
        "url": "https://github.com/AnInsomniacy/tracknet-series-pytorch/"
               "releases/download/model-v2.0.1/tracknet_v2_epoch30.pt",
        "save_path": "models/TrackNet_best.pt",
        "min_size": 100 * 1024 * 1024,  # expected ~136 MB
    },
]


def download_file(url, save_path, min_size):
    """Download with curl, resuming partial files."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cmd = [
        "curl", "-L", "--http1.1", "-C", "-",
        "--retry", "3", "--retry-delay", "3", "--fail",
        "--speed-limit", "10240", "--speed-time", "60",
        "-o", save_path, url,
    ]
    print(f"Downloading {url}\n  -> {save_path} ...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if (res.returncode == 0 and os.path.exists(save_path)
            and os.path.getsize(save_path) >= min_size):
        print(f"Successfully saved to {save_path} "
              f"({os.path.getsize(save_path)} bytes)")
        return True
    msg = (res.stderr or res.stdout or "").strip()
    if msg:
        print(f"curl: {msg.splitlines()[-1]}")
    if os.path.exists(save_path):
        print(f"Partial file at {save_path} "
              f"({os.path.getsize(save_path)} bytes) - rerun to resume.")
    return False


def is_trained_checkpoint(path):
    """True if the file loads as a checkpoint with trained BatchNorm stats."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = getattr(checkpoint, "state_dict", lambda: {})()
    counters = [v for k, v in state_dict.items()
                if k.endswith("num_batches_tracked")]
    return bool(counters) and max(float(v) for v in counters) > 0


def validate_architecture(path):
    """Verify the checkpoint actually fits this project's TrackNet model."""
    try:
        from tracknet_detector import (
            TrackNet, extract_state_dict, remap_reference_state_dict,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(checkpoint)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = remap_reference_state_dict(state_dict)
        model = TrackNet(in_dim=9, out_dim=3)
        model.load_state_dict(state_dict)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def main():
    print("=" * 60)
    print("TrackNet Shuttlecock Weights Downloader")
    print("=" * 60)
    print("Source: AnInsomniacy/tracknet-series-pytorch")
    print("Model:  TrackNetV2, pre-trained on badminton broadcast footage")
    print("Direct link:")
    print("  https://github.com/AnInsomniacy/tracknet-series-pytorch/"
          "releases/download/model-v2.0.1/tracknet_v2_epoch30.pt")
    print("=" * 60)

    for src in WEIGHT_SOURCES:
        path = src["save_path"]
        if os.path.exists(path):
            if (os.path.getsize(path) >= src["min_size"]
                    and is_trained_checkpoint(path)):
                print(f"Found valid TrackNet weights at: {path}")
                return
            print(f"Existing file {path} is invalid (missing, truncated, or "
                  "random-initialized). Re-downloading...")

        if not download_file(src["url"], path, src["min_size"]):
            print("\nAuto-download failed. Please download the weights "
                  "manually in your browser:")
            print("  https://github.com/AnInsomniacy/tracknet-series-pytorch/"
                  "releases/download/model-v2.0.1/tracknet_v2_epoch30.pt")
            print(f"and save the file as: {path}")
            sys.exit(1)

        if not is_trained_checkpoint(path):
            print(f"Downloaded file {path} failed validation - it is not "
                  "trained TrackNet weights. Delete it and run this script "
                  "again, or download manually.")
            sys.exit(1)

        ok, detail = validate_architecture(path)
        if not ok:
            print(f"Downloaded file {path} does not match the TrackNetV2 "
                  f"architecture used here:\n{detail}")
            sys.exit(1)

        print("Weights validated. Run the tracker with:")
        print("  python3 main_screen.py --tracknet")
        return


if __name__ == "__main__":
    main()
