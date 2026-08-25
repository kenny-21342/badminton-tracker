"""
Export the TrackNet model to Apple CoreML (FP16) for fast on-device inference.

On Apple Silicon, CoreML can run the full 512x288 model several times faster
than raw PyTorch on the MPS backend, which makes real-time tracking at full
resolution practical.

NOTE: coremltools ships compiled binaries that only support Python <= 3.13.
This Mac's default interpreter is Python 3.14, so run this script with a
Python 3.12/3.13 interpreter (e.g. python3.12) after installing coremltools
and torch in that environment.

Usage:
    python3 export_tracknet_coreml.py

Output: models/TrackNet_best.mlpackage

The exported package supports 512x288, 384x216 and 256x144 input sizes.
The tracker picks it up automatically (or with --tracknet-engine coreml).
"""

import argparse
import os

import numpy as np
import torch

import coremltools as ct
from tracknet_detector import (
    TrackNet,
    extract_state_dict,
    remap_reference_state_dict,
)


def load_torch_model(model_path):
    model = TrackNet(in_dim=9, out_dim=3)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    if any(k.startswith("backbone.") for k in state_dict):
        state_dict = remap_reference_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="Export TrackNet to CoreML")
    ap.add_argument("--model", default="models/TrackNet_best.pt",
                    help="Source PyTorch checkpoint")
    ap.add_argument("--output", default="models/TrackNet_best.mlpackage",
                    help="Output .mlpackage path")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Checkpoint not found: {args.model}")

    print(f"Loading {args.model} ...")
    model = load_torch_model(args.model)

    # Enumerated input shapes: full res plus the common speed presets.
    shapes = ct.EnumeratedShapes(
        shapes=[[1, 9, 288, 512], [1, 9, 216, 384], [1, 9, 144, 256]]
    )

    print("Tracing model ...")
    example = torch.rand(1, 9, 288, 512)
    traced = torch.jit.trace(model, example)

    print("Converting to CoreML (FP16) ...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="frames", shape=shapes)],
        outputs=[ct.TensorType(name="heatmaps", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        import shutil
        shutil.rmtree(args.output)
    mlmodel.save(args.output)
    print(f"Saved to {args.output}")

    # Sanity check: compare CoreML output with PyTorch on a random input.
    print("Sanity-checking outputs ...")
    x = np.random.rand(1, 9, 288, 512).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    coreml_out = mlmodel.predict({"frames": x})["heatmaps"]
    diff = float(np.abs(torch_out - coreml_out).mean())
    print(f"Mean |PyTorch - CoreML| output difference: {diff:.6f}")
    if diff < 0.01:
        print("OK: CoreML export matches the PyTorch model.")
    else:
        print("WARNING: outputs differ more than expected - verify before use.")


if __name__ == "__main__":
    main()
