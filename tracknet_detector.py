"""
TrackNet shuttlecock detector (Temporal Heatmap Deep Learning Detector).

Uses 3 consecutive video frames to output 2D probability heatmaps for precise
shuttlecock detection without requiring custom dataset labeling.

Paper: TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications
Supports TrackNet / TrackNetV2 / TrackNetV3 pre-trained weights.
"""

import os
import re
import cv2
import collections
import numpy as np
import torch
import torch.nn as nn

# Common default model paths
DEFAULT_MODEL_PATHS = [
    "models/TrackNet_best.pt",
    "weights/TrackNet_best.pt",
    "ckpts/TrackNet_best.pt",
    "TrackNet_best.pt",
    "models/tracknet_model.pt",
    "weights/tracknet_model.pt"
]


class Conv2DBlock(nn.Module):
    """Conv2D + BatchNorm + ReLU block."""
    def __init__(self, in_dim, out_dim):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    """Two Conv2DBlocks in sequence."""
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    """Three Conv2DBlocks in sequence."""
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNet(nn.Module):
    """
    TrackNet PyTorch Architecture (U-Net style MIMO).
    Input: (B, 9, H, W) - 3 consecutive RGB frames concatenated
    Output: (B, 3, H, W) - Heatmaps for each of the 3 input frames
    """
    def __init__(self, in_dim=9, out_dim=3):
        super(TrackNet, self).__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)                                      # (N, 64,  H,   W)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                   # (N, 64,  H/2, W/2)
        x2 = self.down_block_2(x)                                      # (N, 128, H/2, W/2)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                   # (N, 128, H/4, W/4)
        x3 = self.down_block_3(x)                                      # (N, 256, H/4, W/4)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                   # (N, 256, H/8, W/8)
        x = self.bottleneck(x)                                        # (N, 512, H/8, W/8)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)    # (N, 768, H/4, W/4)
        x = self.up_block_1(x)                                         # (N, 256, H/4, W/4)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)    # (N, 384, H/2, W/2)
        x = self.up_block_2(x)                                         # (N, 128, H/2, W/2)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)    # (N, 192, H,   W)
        x = self.up_block_3(x)                                         # (N, 64,  H,   W)
        x = self.predictor(x)                                          # (N, 3,   H,   W)
        x = self.sigmoid(x)                                            # (N, 3,   H,   W)
        return x


# ---------------------------------------------------------------------------
# Checkpoint compatibility helpers
# ---------------------------------------------------------------------------
# TrackNetV2 checkpoints from the open "TrackNet Series PyTorch" project
# (github.com/AnInsomniacy/tracknet-series-pytorch) use a different module
# naming (backbone.enc1.0.conv.weight ...) for the exact same architecture as
# the TrackNet class above, so we remap the keys at load time.

_REF_BLOCK_TO_PROJECT = {
    "enc1": "down_block_1",
    "enc2": "down_block_2",
    "enc3": "down_block_3",
    "enc4": "bottleneck",
    "dec1": "up_block_1",
    "dec2": "up_block_2",
    "dec3": "up_block_3",
}
_REF_CONV_KEY_RE = re.compile(
    r"^backbone\.(enc[1-4]|dec[1-3])\.(\d+)\."
    r"(conv\.weight|bn\.(?:weight|bias|running_mean|running_var|num_batches_tracked))$"
)
_REF_SEQ_KEY_RE = re.compile(
    r"^backbone\.(enc[1-4]|dec[1-3])\.(\d+)\.(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$"
)
_REF_OUTPUT_KEY_RE = re.compile(r"^backbone\.output\.(weight|bias)$")


def remap_reference_state_dict(state_dict):
    """Rename TrackNetV2 reference keys to this project's module naming."""
    remapped = {}
    for key, value in state_dict.items():
        m = _REF_CONV_KEY_RE.match(key)
        if m:
            block, index, suffix = m.groups()
            remapped[f"{_REF_BLOCK_TO_PROJECT[block]}.conv_{int(index) + 1}.{suffix}"] = value
            continue
        m = _REF_SEQ_KEY_RE.match(key)
        if m:
            # Reference uses nn.Sequential(Conv2d, BatchNorm2d, ReLU), so the
            # inner module index is 0 for the conv and 1 for the batchnorm.
            block, layer, module_idx, suffix = m.groups()
            inner = "conv" if module_idx == "0" else "bn"
            remapped[
                f"{_REF_BLOCK_TO_PROJECT[block]}.conv_{int(layer) + 1}.{inner}.{suffix}"
            ] = value
            continue
        m = _REF_OUTPUT_KEY_RE.match(key)
        if m:
            remapped[f"predictor.{m.group(1)}"] = value
            continue
        remapped[key] = value
    return remapped


def extract_state_dict(checkpoint):
    """Return a state dict from any torch.load() result we accept."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    raise ValueError(
        f"Unsupported checkpoint format: {type(checkpoint).__name__}. "
        "Expected a torch.save() dict containing a model state_dict."
    )


def assert_trained_weights(state_dict, model_path):
    """Reject checkpoints that were never trained (BatchNorm counters == 0)."""
    counters = [v for k, v in state_dict.items() if k.endswith("num_batches_tracked")]
    if not counters:
        return
    if max(float(v) for v in counters) <= 0:
        raise RuntimeError(
            f"Checkpoint {model_path} appears to be randomly initialized "
            "(all BatchNorm training counters are 0), so it cannot detect "
            "shuttlecocks. Download real pre-trained weights with:\n"
            "  python3 download_tracknet.py"
        )


def heatmap_to_point(heatmap, confidence, w_orig, h_orig,
                     input_width, input_height, upsample_heatmap=True):
    """
    Convert a TrackNet heatmap into a (cx, cy) in full-frame coordinates.

    When upsample_heatmap is True (default), the heatmap is first upsampled to
    the original frame resolution before thresholding and centroid extraction.
    This costs <1 ms but recovers near full-resolution localization precision
    even when inference runs at a reduced input size.
    """
    hm = heatmap
    if upsample_heatmap:
        hm = cv2.resize(heatmap, (w_orig, h_orig),
                        interpolation=cv2.INTER_CUBIC)

    binary = (hm >= confidence).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            x_peak = M["m10"] / M["m00"]
            y_peak = M["m01"] / M["m00"]
        else:
            y_peak, x_peak = np.unravel_index(np.argmax(hm), hm.shape)
    else:
        y_peak, x_peak = np.unravel_index(np.argmax(hm), hm.shape)

    if upsample_heatmap:
        return int(round(x_peak)), int(round(y_peak))
    return (int(round(x_peak * (w_orig / float(input_width)))),
            int(round(y_peak * (h_orig / float(input_height)))))


class TrackNetDetector:
    """
    Real-time temporal TrackNet detector wrapper.
    Processes consecutive video frames using PyTorch TrackNet weights.
    """
    def __init__(self, model_path=None, confidence=0.5, input_width=512,
                 input_height=288, device=None, fp16=False,
                 upsample_heatmap=True):
        self.confidence = confidence
        self.input_width = input_width
        self.input_height = input_height
        self.upsample_heatmap = upsample_heatmap

        # Setup compute device (mps for Apple Silicon, cuda, or cpu)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Resolve model path
        if model_path is None:
            for p in DEFAULT_MODEL_PATHS:
                if os.path.exists(p):
                    model_path = p
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"TrackNet model weights file not found. Checked: {DEFAULT_MODEL_PATHS}.\n"
                f"Please download pre-trained weights (e.g. TrackNet_best.pt) to models/"
            )

        self.model_path = model_path
        self.model = TrackNet(in_dim=9, out_dim=3)

        # Load weights on CPU first, then move to the compute device. This
        # avoids MPS/optimizer-state quirks in third-party checkpoints.
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(checkpoint)

        # Strip DataParallel/DDP prefixes if present.
        if (any(k.startswith("module.") for k in state_dict)
                and not any(k.startswith("module.") for k in self.model.state_dict())):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        # Remap TrackNetV2 reference keys if needed.
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = remap_reference_state_dict(state_dict)
            print(f"[TrackNet] Remapped reference checkpoint keys: {model_path}")

        assert_trained_weights(state_dict, model_path)

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {model_path} is not compatible with the TrackNet "
                "architecture in this project (TrackNetV2 U-Net: 9 input "
                f"channels, 3 sigmoid heatmaps). Details:\n{exc}"
            ) from exc

        self.model.to(self.device)
        self.fp16 = fp16
        if fp16:
            self.model = self.model.half()
            print("[TrackNet] Running in float16 mode")
        self.model.eval()

        # 3-frame buffer for temporal input
        self.frame_buffer = collections.deque(maxlen=3)
        self.last_confidence = 0.0
        self.last_heatmap = None

    def reset(self):
        """Reset frame buffer and state."""
        self.frame_buffer.clear()
        self.last_confidence = 0.0
        self.last_heatmap = None

    def detect(self, frame: np.ndarray):
        """
        Processes frame and returns (cx, cy) in full-frame coordinates, or None.
        """
        h_orig, w_orig = frame.shape[:2]

        # Resize to TrackNet input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Fill frame buffer
        if len(self.frame_buffer) == 0:
            # Cold start: replicate initial frame 3 times
            for _ in range(3):
                self.frame_buffer.append(rgb)
        else:
            self.frame_buffer.append(rgb)

        # Stack 3 frames along channel axis -> (H, W, 9) -> (9, H, W)
        frames_stacked = np.concatenate(list(self.frame_buffer), axis=2)
        dtype = torch.float16 if self.fp16 else torch.float32
        tensor_in = torch.from_numpy(frames_stacked).permute(2, 0, 1).unsqueeze(0)
        tensor_in = tensor_in.to(device=self.device, dtype=dtype)

        with torch.no_grad():
            heatmaps = self.model(tensor_in)  # (1, 3, H, W)
            # Use current frame heatmap (channel index 2, or last frame)
            heatmap = heatmaps[0, -1].cpu().float().numpy()

        self.last_heatmap = heatmap
        max_val = float(heatmap.max())
        self.last_confidence = max_val

        if max_val < self.confidence:
            return None

        return heatmap_to_point(heatmap, self.confidence, w_orig, h_orig,
                                self.input_width, self.input_height,
                                self.upsample_heatmap)


class TrackNetHybridDetector:
    """
    Hybrid detector combining TrackNet as primary and Classic CV as fallback.
    """
    def __init__(self, model_path=None, confidence=0.5, classic_scale=0.5,
                 enable_fallback=True, device=None, input_width=512,
                 input_height=288, fp16=False, engine="auto",
                 upsample_heatmap=True):
        self.tracknet = None
        self.enable_fallback = enable_fallback
        self.classic_scale = classic_scale
        self.engine_name = "none"

        common = dict(
            model_path=model_path,
            confidence=confidence,
            input_width=input_width,
            input_height=input_height,
            upsample_heatmap=upsample_heatmap,
        )

        # Prefer MLX (2x faster at reduced resolutions on Apple Silicon)
        # unless the user forced torch.
        if engine in ("auto", "mlx"):
            try:
                from mlx_tracknet_detector import MLXTrackNetDetector
                self.tracknet = MLXTrackNetDetector(**common, fp16=fp16)
                self.engine_name = "mlx"
                print(f"Loaded TrackNet (MLX engine) from: "
                      f"{self.tracknet.model_path}")
            except ImportError:
                if engine == "mlx":
                    print("Notice: MLX engine unavailable (install with: "
                          "pip install mlx). Falling back to PyTorch.")
                self.tracknet = None
            except Exception as e:
                if engine == "mlx":
                    print(f"Notice: MLX TrackNet could not be initialized ({e})")
                self.tracknet = None

        if self.tracknet is None and engine in ("auto", "torch"):
            try:
                self.tracknet = TrackNetDetector(
                    **common, device=device, fp16=fp16)
                self.engine_name = "torch"
                print(f"Loaded TrackNet (PyTorch engine) from: "
                      f"{self.tracknet.model_path} ({self.tracknet.device})")
            except Exception as e:
                print(f"Notice: TrackNet could not be initialized ({e})")
                self.tracknet = None

        # Always initialize Classic CV fallback
        from detector import ShuttlecockDetector
        self.classic = ShuttlecockDetector(scale=classic_scale)

        if self.tracknet is None:
            print("TrackNet engine unavailable - using classic CV only.")

    def reset(self):
        if self.tracknet:
            self.tracknet.reset()
        self.classic.reset()

    def detect(self, frame: np.ndarray):
        """
        Returns (point, source_str) in full-frame coordinates.
        point is (x, y) or None.
        source_str is 'tracknet', 'classic', or 'miss'.
        """
        # Try TrackNet first
        if self.tracknet is not None:
            pt = self.tracknet.detect(frame)
            if pt is not None:
                return pt, "tracknet"

        # Fallback to classic CV detector
        if self.enable_fallback:
            det = self.classic.detect(frame)
            if det is not None:
                inv = 1.0 / self.classic.scale
                return (int(round(det[0] * inv)), int(round(det[1] * inv))), "classic"

        return None, "miss"
