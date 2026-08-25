"""
YOLO-based shuttlecock detector.

Wraps ultralytics YOLOv8 for real-time shuttlecock detection on screen-
captured frames. Designed to be a drop-in replacement for the classic
ShuttlecockDetector in detector.py when a trained model is available.

Workflow:
  1. python3 extract_frames.py          # grab frames from match video/screen
  2. python3 labeler.py                  # click-drag boxes around the shuttle
  3. python3 train.py                    # fine-tune YOLOv8 on your labels
  4. python3 main_screen.py --yolo       # run tracker with trained model

Until you train, it falls back to the classic detector automatically.
"""
import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


DEFAULT_MODEL_PATH = "runs/detect/train/weights/best.pt"


class YOLODetector:
    """
    Real-time shuttlecock detector using a fine-tuned YOLOv8-nano model.

    Unlike the classic CV detector, this doesn't rely on brightness
    thresholds or motion masks — it learns shuttle appearance directly
    (including blurred fast-moving shuttles) and naturally ignores
    jerseys, court lines, and scorebugs.

    The detector returns the shuttle position in *full-frame* coordinates
    (no downscaled coordinate space — YOLO handles full resolution).
    """

    def __init__(self, model_path=None, confidence=0.25, imgsz=640,
                 device=None):
        """
        Args:
            model_path:  Path to .pt weights. None = try default trained path.
            confidence:  Minimum detection confidence (0-1). Lower = more
                         sensitive but more false positives.
            imgsz:       Inference image size (pixels). 640 is default; 480
                         is faster on CPU at slight accuracy cost.
            device:      'cpu', 'mps', 'cuda', or None (auto-detect).
        """
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics")

        # Resolve model path
        if model_path is None:
            # Try several common locations
            candidates = [
                DEFAULT_MODEL_PATH,
                "models/shuttlecock_yolov8n.pt",
                "shuttlecock_best.pt",
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found. Expected one of:\n"
                f"  {DEFAULT_MODEL_PATH}\n"
                f"  models/shuttlecock_yolov8n.pt\n"
                f"  shuttlecock_best.pt\n"
                f"Train one first with: python3 train.py\n"
                f"Or download a pre-trained model and pass its path.")

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.imgsz = imgsz

        # Auto-detect MPS on Apple Silicon if device not specified.
        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
            except Exception:
                pass
        self.device = device or "cpu"

        # Try to find the shuttlecock class in the model.
        # We support any class name containing "shuttle" or "birdie".
        self.shuttle_class_ids = []
        for cid, name in self.model.names.items():
            name_lower = str(name).lower()
            if "shuttle" in name_lower or "bird" in name_lower:
                self.shuttle_class_ids.append(cid)
        # If none found, use class 0 (common for single-class datasets).
        if not self.shuttle_class_ids:
            self.shuttle_class_ids = [0]

        # --- Tracking state (for gap filling / velocity) ---
        self.last_point = None
        self.last_velocity = (0.0, 0.0)
        self.confidence_smoothed = 0.0
        self.lost_count = 0
        self.alpha_vel = 0.4

        # History for physics-based prediction (parabolic trajectory).
        self.history = []            # recent (x, y) positions
        self.max_history = 8         # frames of history to keep
        self.gravity_px = 0.0        # estimated gravity in px/frame²

    def detect(self, frame):
        """
        Run YOLO inference on a frame and return the shuttlecock position.

        Returns:
            (x, y) in full-frame pixel coords, or None if not detected.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            stream=True,           # avoid per-call memory alloc overhead
            classes=self.shuttle_class_ids,
        )
        result = next(results)      # single image → single result

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self.last_box = None
            self.last_conf = 0.0
            return self._on_miss()

        # Pick the highest-confidence detection.
        best_idx = int(boxes.conf.argmax())
        box = boxes[best_idx]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        conf = float(box.conf[0])

        # Store for debug drawing.
        self.last_box = (int(x1), int(y1), int(x2), int(y2))
        self.last_conf = conf

        # Update velocity estimate.
        if self.last_point is not None:
            new_vx = cx - self.last_point[0]
            new_vy = cy - self.last_point[1]
            a = self.alpha_vel
            self.last_velocity = (
                self.last_velocity[0] * (1 - a) + new_vx * a,
                self.last_velocity[1] * (1 - a) + new_vy * a,
            )

        self.last_point = (cx, cy)
        self.confidence_smoothed = conf
        self.lost_count = 0

        # Update history and estimate gravity (vertical acceleration).
        self.history.append((cx, cy))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self._estimate_gravity()

        return (cx, cy)

    def _estimate_gravity(self):
        """
        Estimate vertical acceleration from position history.
        In badminton footage, gravity pulls the shuttle down consistently.
        We compute the second derivative of y-position across the last
        few frames and smooth it. This lets gap-filling follow parabolic
        trajectories instead of straight lines.
        """
        if len(self.history) < 5:
            return
        # Compute per-frame velocity, then acceleration (delta velocity).
        pts = self.history[-5:]  # use last 5 positions
        vels_y = [pts[i+1][1] - pts[i][1] for i in range(len(pts)-1)]
        accs_y = [vels_y[i+1] - vels_y[i] for i in range(len(vels_y)-1)]
        if accs_y:
            # Smooth gravity estimate (positive = downward accel).
            raw_g = sum(accs_y) / len(accs_y)
            self.gravity_px = self.gravity_px * 0.7 + raw_g * 0.3

    def _predict_next(self, steps):
        """
        Physics-based position prediction: parabolic trajectory.
        x(t) = x0 + vx*t           (horizontal: constant velocity)
        y(t) = y0 + vy*t + 0.5*g*t² (vertical: gravity acceleration)

        This is what makes the gap-fill follow cross-court smashes
        instead of predicting straight lines.
        """
        if self.last_point is None:
            return None
        vx, vy = self.last_velocity
        t = steps
        px = self.last_point[0] + vx * t
        py = self.last_point[1] + vy * t + 0.5 * self.gravity_px * t * t
        return (int(px), int(py))

    def _on_miss(self):
        """No YOLO detection this frame — return None (no prediction)."""
        self.last_box = None
        self.last_conf = 0.0
        self.lost_count += 1
        if self.lost_count > 10:
            self.last_point = None
            self.last_velocity = (0.0, 0.0)
            self.history = []
        return None

    def reset(self):
        """Clear tracking state."""
        self.last_point = None
        self.last_velocity = (0.0, 0.0)
        self.confidence_smoothed = 0.0
        self.lost_count = 0
        self.history = []
        self.gravity_px = 0.0

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, v):
        self._confidence = v


class HybridDetector:
    """
    Combines YOLO and the classic CV detector for maximum robustness.

    Strategy:
      - Primary:   YOLO (learned shuttle appearance — robust to jerseys, etc.)
      - Fallback:  Classic CV detector when YOLO finds nothing (bridge gaps
                   where the shuttle is too blurred/small for YOLO but still
                   visible as a bright moving blob).

    This lets you run immediately with YOLO, and if the model occasionally
    loses the shuttle (e.g. extreme motion blur), the classic detector
    fills the gap.
    """

    def __init__(self, yolo_path=None, classic_scale=0.5,
                 confidence=0.25, enable_fallback=True,
                 imgsz=480, device=None):
        from detector import ShuttlecockDetector

        self.yolo = None
        self.classic = None
        self.enable_fallback = enable_fallback
        self.last_source = "none"

        # Try to load YOLO
        try:
            self.yolo = YOLODetector(model_path=yolo_path, confidence=confidence,
                                    imgsz=imgsz, device=device)
            print(f"[hybrid] YOLO detector loaded successfully")
        except (FileNotFoundError, ImportError) as e:
            print(f"[hybrid] YOLO not available: {e}")
            if not enable_fallback:
                raise

        # Always have classic as backup
        self.classic = ShuttlecockDetector(scale=classic_scale)
        if self.yolo is None:
            print(f"[hybrid] Using classic CV detector only")
        else:
            print(f"[hybrid] Classic CV fallback {'enabled' if enable_fallback else 'disabled'}")

    def detect(self, frame):
        """
        Returns ((x, y) or None, source_str).
        Coordinates are in full-frame space.
        """
        # Try YOLO first.
        if self.yolo is not None:
            pt = self.yolo.detect(frame)
            if pt is not None:
                self.last_source = "yolo"
                return pt, "yolo"

        # Fallback to classic CV.
        if self.enable_fallback or self.yolo is None:
            det = self.classic.detect(frame)
            if det is not None:
                # Classic detector returns downscaled coords; convert.
                inv = 1.0 / self.classic.scale
                pt = (int(det[0] * inv), int(det[1] * inv))
                self.last_source = "classic"
                return pt, "classic"

        self.last_source = "miss"
        return None, "miss"

    def reset(self):
        if self.yolo:
            self.yolo.reset()
        self.classic.reset()
