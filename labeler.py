"""
Interactive labeling tool for shuttlecock detection.

Controls:
    Left click/drag   Draw box around shuttle (auto-advances to next)
    Right click       No shuttle (empty label, auto-advances)
    B                 Previous frame (edit earlier labels)
    D                 Delete current box (stay on frame)
    TAB               Switch mode: drag → click(30px) → point(tiny)
    Q                 Quit (progress saved)

Usage:
    python3 labeler.py                      # drag mode
    python3 labeler.py --box-size 30        # click mode
    python3 labeler.py --images my_frames/  # custom directory
"""
import cv2
import os
import glob
import argparse

from extract_frames import is_court_view


class Labeler:
    def __init__(self, images_dir, labels_dir, box_size=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.box_size = box_size
        os.makedirs(labels_dir, exist_ok=True)

        self.images = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg")) +
            glob.glob(os.path.join(images_dir, "*.png"))
        )
        if not self.images:
            print(f"No images found in {images_dir}/")
            return

        self.idx = self._first_unlabeled()
        self.drawing = False
        self.start_pt = None
        self.end_pt = None
        self.current_box = None
        self.window = "Shuttlecock Labeler"
        self.mode = "click" if box_size else "drag"
        self._disp_scale = 1.0

    def _first_unlabeled(self):
        for i, img_path in enumerate(self.images):
            if not os.path.exists(self._label_path(img_path)):
                return i
        return 0

    def _label_path(self, img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.labels_dir, base + ".txt")

    def _save_label(self, img_path, box):
        label_path = self._label_path(img_path)
        img_h, img_w = self._img_shape
        if box is None:
            open(label_path, "w").close()
            return
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = abs(x2 - x1) / img_w
        bh = abs(y2 - y1) / img_h
        with open(label_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _load_existing(self, img_path):
        label_path = self._label_path(img_path)
        if not os.path.exists(label_path):
            return None
        with open(label_path) as f:
            line = f.readline().strip()
        if not line:
            return None
        parts = line.split()
        _, cx, cy, bw, bh = map(float, parts)
        img_h, img_w = self._img_shape
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        return (x1, y1, x2, y2)

    def _advance(self):
        """Save current box and go to next frame."""
        img_path = self.images[self.idx]
        self._save_label(img_path, self.current_box)
        self.idx += 1

    def _mark_empty(self):
        """Save empty label and go to next frame."""
        img_path = self.images[self.idx]
        self._save_label(img_path, None)
        self.idx += 1

    def _on_mouse(self, event, x, y, flags, param):
        real_x = int(x / self._disp_scale)
        real_y = int(y / self._disp_scale)

        # Right click = no shuttle, auto-advance.
        if event == cv2.EVENT_RBUTTONDOWN:
            self._mark_empty()
            return

        if self.mode == "click":
            half = self.box_size // 2
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_box = (real_x - half, real_y - half,
                                    real_x + half, real_y + half)
                self._advance()
        elif self.mode == "point":
            half = 8
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_box = (real_x - half, real_y - half,
                                    real_x + half, real_y + half)
                self._advance()
        else:  # drag
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_pt = (real_x, real_y)
                self.end_pt = (real_x, real_y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.end_pt = (real_x, real_y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.end_pt = (real_x, real_y)
                if self.start_pt and self.end_pt:
                    x1 = min(self.start_pt[0], self.end_pt[0])
                    y1 = min(self.start_pt[1], self.end_pt[1])
                    x2 = max(self.start_pt[0], self.end_pt[0])
                    y2 = max(self.start_pt[1], self.end_pt[1])
                    if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                        self.current_box = (x1, y1, x2, y2)
                        self._advance()

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

        total = len(self.images)
        print(f"\nLabeling {total} images from {self.images_dir}/")
        print(f"Mode: {self.mode}  |  Starting at frame {self.idx}")
        print("Left click/drag = box + auto-next")
        print("Right click     = no shuttle + auto-next")
        print("B=back  D=delete box  TAB=switch mode  Q=quit\n")

        while 0 <= self.idx < total:
            img_path = self.images[self.idx]
            img = cv2.imread(img_path)
            self._img_shape = img.shape[:2]  # (h, w, channels)

            court = is_court_view(img)

            disp_h = 900
            scale = 1.0
            if img.shape[0] > disp_h:
                scale = disp_h / img.shape[0]
                disp = cv2.resize(img, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
            else:
                disp = img.copy()
            self._disp_scale = scale

            self.drawing = False
            self.start_pt = None
            self.end_pt = None
            self.current_box = self._load_existing(img_path)

            while True:
                # If index changed (via mouse auto-advance), break to reload.
                if self.idx >= total or self.images[self.idx] != img_path:
                    break

                canvas = disp.copy()
                s = scale

                if self.current_box:
                    b = self.current_box
                    cv2.rectangle(canvas,
                                   (int(b[0] * s), int(b[1] * s)),
                                   (int(b[2] * s), int(b[3] * s)),
                                   (0, 255, 0), 2)
                if self.drawing and self.start_pt and self.end_pt:
                    cv2.rectangle(canvas,
                                   (int(self.start_pt[0] * s),
                                    int(self.start_pt[1] * s)),
                                   (int(self.end_pt[0] * s),
                                    int(self.end_pt[1] * s)),
                                   (0, 255, 255), 1)

                labeled = sum(
                    1 for im in self.images
                    if os.path.exists(self._label_path(im)))
                cv2.rectangle(canvas, (0, 0),
                              (canvas.shape[1], 40), (0, 0, 0), -1)
                view_tag = "COURT" if court else "CLOSE-UP"
                mode_tag = f"MODE:{self.mode.upper()}"
                status = "LABELED" if (self.current_box or
                    (os.path.exists(self._label_path(img_path)) and
                     os.path.getsize(self._label_path(img_path)) == 0)) else "EMPTY"
                hud = (f"[{self.idx+1}/{total}]  {labeled}/{total} labeled  "
                       f"{mode_tag}  {view_tag}  {status}")
                cv2.putText(canvas, hud, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255) if court else (0, 200, 255), 1)

                cv2.imshow(self.window, canvas)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    self.idx = -1
                    break
                elif key == ord("b"):
                    self.idx = max(0, self.idx - 1)
                    break
                elif key == ord("d"):
                    self.current_box = None
                    label_path = self._label_path(img_path)
                    if os.path.exists(label_path):
                        os.remove(label_path)
                elif key == ord("n"):
                    self._mark_empty()
                    break
                elif key == ord("t") or key == 9:  # TAB
                    if self.mode == "drag":
                        self.mode = "click"
                        self.box_size = 30
                    elif self.mode == "click":
                        self.mode = "point"
                    else:
                        self.mode = "drag"
                    print(f"  [mode = {self.mode}]")
                elif key in (13, 32):  # Enter / Space = save + next
                    self._advance()
                    break

        cv2.destroyAllWindows()
        labeled = sum(1 for im in self.images
                      if os.path.exists(self._label_path(im)))
        print(f"\nDone: {labeled}/{total} images labeled.")


def main():
    ap = argparse.ArgumentParser(description="Shuttlecock labeling tool")
    ap.add_argument("--images", type=str, default="datasets/images",
                    help="Directory of images to label")
    ap.add_argument("--labels", type=str, default="datasets/labels",
                    help="Directory for YOLO label files")
    ap.add_argument("--box-size", type=int, default=None,
                    help="Fixed square box size (click mode).")
    args = ap.parse_args()

    labeler = Labeler(args.images, args.labels, args.box_size)
    if labeler.images:
        labeler.run()


if __name__ == "__main__":
    main()
