"""
TrackNet detector running on MLX (Apple Silicon GPU).

MLX is Apple's machine-learning framework for Apple Silicon. For the same
TrackNet model it runs roughly 2x faster than PyTorch's MPS backend at reduced
resolutions, and it works in-process (no separate interpreter needed).

This module mirrors the interface of tracknet_detector.TrackNetDetector so it
can be swapped in by TrackNetHybridDetector(engine="mlx").
"""

import collections
import os

import cv2
import numpy as np
import torch

from tracknet_detector import (
    DEFAULT_MODEL_PATHS,
    assert_trained_weights,
    extract_state_dict,
    heatmap_to_point,
    remap_reference_state_dict,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_AVAILABLE = True
    _MLX_IMPORT_ERROR = None
except ImportError as _mlx_error:
    _MLX_AVAILABLE = False
    # Keep the reason. "mlx is not installed" is only one way this fails; in a
    # frozen bundle mlx is present but a submodule or its Metal library is
    # missing, and that needs a different fix.
    _MLX_IMPORT_ERROR = _mlx_error


_POOL = nn.MaxPool2d(2, 2) if _MLX_AVAILABLE else None


def _build_mlx_params(state_dict, dtype):
    """Convert a PyTorch state dict into MLX arrays (NHWC weight layout)."""
    params = {}
    for key, value in state_dict.items():
        if key.endswith("num_batches_tracked"):
            continue
        arr = mx.array(value.numpy()).astype(dtype)
        if key.endswith("conv.weight") or key == "predictor.weight":
            # torch (C_out, C_in, KH, KW) -> mlx (C_out, KH, KW, C_in)
            arr = arr.transpose(0, 2, 3, 1)
        params[key] = arr
    return params


def _conv(x, w):
    return mx.conv2d(x, w, stride=1, padding=1)


def _bn(x, p, key):
    w = p[key + ".bn.weight"][None, None, None, :]
    b = p[key + ".bn.bias"][None, None, None, :]
    m = p[key + ".bn.running_mean"][None, None, None, :]
    v = p[key + ".bn.running_var"][None, None, None, :]
    return (x - m) / mx.sqrt(v + 1e-5) * w + b


def _conv_bn(x, p, key):
    return nn.relu(_bn(_conv(x, p[key + ".conv.weight"]), p, key))


def _block(x, p, prefix, n):
    for i in range(1, n + 1):
        x = _conv_bn(x, p, f"{prefix}.conv_{i}")
    return x


def _up2(x):
    """Nearest 2x upsample (matches torch nn.Upsample nearest)."""
    return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)


def _forward(params, x):
    """TrackNet forward pass. x is NHWC (1, H, W, 9); output NHWC (1, H, W, 3)."""
    x1 = _block(x, params, "down_block_1", 2)
    x = _POOL(x1)
    x2 = _block(x, params, "down_block_2", 2)
    x = _POOL(x2)
    x3 = _block(x, params, "down_block_3", 3)
    x = _POOL(x3)
    x = _block(x, params, "bottleneck", 3)
    x = _up2(x)
    x = _block(mx.concatenate([x, x3], axis=3), params, "up_block_1", 3)
    x = _up2(x)
    x = _block(mx.concatenate([x, x2], axis=3), params, "up_block_2", 2)
    x = _up2(x)
    x = _block(mx.concatenate([x, x1], axis=3), params, "up_block_3", 2)
    y = mx.conv2d(x, params["predictor.weight"], stride=1, padding=0)
    return mx.sigmoid(y + params["predictor.bias"][None, None, None, :])


class MLXTrackNetDetector:
    """
    MLX-accelerated TrackNet detector, drop-in for TrackNetDetector.
    Coordinates returned are in full-frame space.
    """

    def __init__(self, model_path=None, confidence=0.5, input_width=512,
                 input_height=288, fp16=True, upsample_heatmap=True,
                 device=None):
        if not _MLX_AVAILABLE:
            raise ImportError(
                f"MLX could not be imported: {_MLX_IMPORT_ERROR}. "
                "If mlx is missing, run: pip install mlx")

        self.confidence = confidence
        self.input_width = input_width
        self.input_height = input_height
        self.upsample_heatmap = upsample_heatmap
        self.device = "mlx"
        self.fp16 = fp16

        if model_path is None:
            for p in DEFAULT_MODEL_PATHS:
                if os.path.exists(p):
                    model_path = p
                    break
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"TrackNet model weights file not found. Checked: "
                f"{DEFAULT_MODEL_PATHS}.\n"
                f"Please download pre-trained weights (e.g. TrackNet_best.pt) "
                f"to models/")
        self.model_path = model_path

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(ckpt)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = remap_reference_state_dict(state_dict)
            print(f"[TrackNet/MLX] Remapped reference checkpoint keys: "
                  f"{model_path}")
        assert_trained_weights(state_dict, model_path)

        dtype = mx.float16 if fp16 else mx.float32
        self.params = _build_mlx_params(state_dict, dtype)

        self.frame_buffer = collections.deque(maxlen=3)
        self.last_confidence = 0.0
        self.last_heatmap = None

    def reset(self):
        self.frame_buffer.clear()
        self.last_confidence = 0.0
        self.last_heatmap = None

    def detect(self, frame):
        h_orig, w_orig = frame.shape[:2]

        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if len(self.frame_buffer) == 0:
            for _ in range(3):
                self.frame_buffer.append(rgb)
        else:
            self.frame_buffer.append(rgb)

        # NHWC: (H, W, 9) - MLX convolutions are channel-last.
        frames_stacked = np.concatenate(list(self.frame_buffer), axis=2)
        dtype = mx.float16 if self.fp16 else mx.float32
        tensor_in = mx.array(frames_stacked).astype(dtype)[None, ...]

        heatmaps = _forward(self.params, tensor_in)
        mx.eval(heatmaps)
        heatmap = np.array(heatmaps)[0, :, :, 2].astype(np.float32)

        self.last_heatmap = heatmap
        max_val = float(heatmap.max())
        self.last_confidence = max_val

        if max_val < self.confidence:
            return None

        return heatmap_to_point(heatmap, self.confidence, w_orig, h_orig,
                                self.input_width, self.input_height,
                                self.upsample_heatmap)
