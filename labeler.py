"""
Interactive labeling tool for shuttlecock detection.

Controls:
    Left click/drag   Draw box around shuttle (auto-advances to next)
    Right click       No shuttle (empty label, auto-advances)
    A / SPACE         Accept the suggested box (auto-advances)
    B                 Previous frame (edit earlier labels)
    D                 Delete current box (stay on frame)
    TAB               Switch mode: drag → click(30px) → point(tiny)
    Q                 Quit (progress saved)

Usage:
    python3 labeler.py                      # drag mode
    python3 labeler.py --box-size 30        # click mode
    python3 labeler.py --images my_frames/  # custom directory
    python3 labeler.py --no-suggest         # draw everything by hand
    python3 labeler.py --stride 3           # label every 3rd frame, fill rest

A suggested box is drawn in amber and is NOT a label until you accept it. On
the measured clips two thirds of frames are one keypress this way -- see
label_assist.py for the numbers and for what `--stride` costs in accuracy.
"""
import cv2
import os
import glob
import argparse

from extract_frames import is_court_view
import label_assist


class Labeler:
    def __init__(self, images_dir, labels_dir, box_size=None,
                 suggest=True, stride=1, model_path=None):
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

        self.stride = max(1, stride)
        self.anchors = label_assist.anchor_indices(self.images, self.stride)
        self.provenance = label_assist.load_sidecar(labels_dir)
        self.accepted_this_run = 0
        self.drawn_this_run = 0

        self.suggestions = {}
        if suggest:
            self.suggestions = self._load_suggestions(model_path)
        self.suggestion = None

        self.idx = self._first_unlabeled()
        self.drawing = False
        self.start_pt = None
        self.end_pt = None
        self.current_box = None
        self.window = "Shuttlecock Labeler"
        self.mode = "click" if box_size else "drag"
        self._disp_scale = 1.0

    def _load_suggestions(self, model_path):
        """Pre-compute detector suggestions, or explain why there are none."""
        if not label_assist.is_contiguous(self.images):
            print("Note: these frames are not consecutive footage, so TrackNet "
                  "has no motion to read and its suggestions would be "
                  "meaningless. Labelling by hand.")
            return {}
        try:
            cache = os.path.join(self.labels_dir, "_suggestions.json")
            return label_assist.compute_suggestions(
                self.images, self.box_size or 30, cache, model_path)
        except Exception as error:
            print(f"Note: suggestions unavailable ({error}). Labelling by hand.")
            return {}

    def _is_anchor(self, index):
        """Whether this frame is one the human is asked to label."""
        return index in self.anchors

    def _first_unlabeled(self):
        for i, img_path in enumerate(self.images):
            if not self._is_anchor(i):
                continue
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

    def _next_anchor(self, after):
        """The next frame the human is asked to label."""
        for i in range(after + 1, len(self.images)):
            if self._is_anchor(i):
                return i
        return len(self.images)

    def _advance(self, source="drawn"):
        """Save current box and go to the next frame to label."""
        img_path = self.images[self.idx]
        self._save_label(img_path, self.current_box)
        self._record(img_path, source)
        if source == "accepted":
            self.accepted_this_run += 1
        else:
            self.drawn_this_run += 1
        self.idx = self._next_anchor(self.idx)

    def _accept_suggestion(self):
        """Take the suggested box as the label for this frame."""
        if not self.suggestion:
            return
        self.current_box = self.suggestion
        self._advance(source="accepted")

    def _record(self, img_path, source):
        """Note how this label came to exist, for the sidecar."""
        name = os.path.basename(img_path)
        for key in ("accepted", "filled"):
            if name in self.provenance[key]:
                self.provenance[key].remove(name)
        if source == "accepted":
            self.provenance["accepted"].append(name)

    def _mark_empty(self):
        """Save empty label and go to the next frame to label."""
        img_path = self.images[self.idx]
        self._save_label(img_path, None)
        self._record(img_path, "drawn")
        self.drawn_this_run += 1
        self.idx = self._next_anchor(self.idx)

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
            # A suggestion is only offered where nothing has been decided yet,
            # so revisiting a frame with B shows what you chose, not what the
            # detector wanted.
            already = os.path.exists(self._label_path(img_path))
            self.suggestion = (
                None if already
                else self.suggestions.get(os.path.basename(img_path)))

            while True:
                # If index changed (via mouse auto-advance), break to reload.
                if self.idx >= total or self.images[self.idx] != img_path:
                    break

                canvas = disp.copy()
                s = scale

                if self.suggestion and not self.current_box:
                    b = self.suggestion
                    # Amber and thin, against the committed label's solid
                    # green: nothing on screen should look saved until it is.
                    cv2.rectangle(canvas,
                                   (int(b[0] * s), int(b[1] * s)),
                                   (int(b[2] * s), int(b[3] * s)),
                                   (0, 180, 255), 1)
                    cv2.putText(canvas, "SUGGESTED - [a] accept",
                                (int(b[0] * s), max(12, int(b[1] * s) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)
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
                    1 for i, im in enumerate(self.images)
                    if self._is_anchor(i)
                    and os.path.exists(self._label_path(im)))
                cv2.rectangle(canvas, (0, 0),
                              (canvas.shape[1], 40), (0, 0, 0), -1)
                view_tag = "COURT" if court else "CLOSE-UP"
                mode_tag = f"MODE:{self.mode.upper()}"
                status = "LABELED" if (self.current_box or
                    (os.path.exists(self._label_path(img_path)) and
                     os.path.getsize(self._label_path(img_path)) == 0)) else "EMPTY"
                to_label = len(self.anchors)
                stride_tag = (f"  STRIDE:{self.stride}"
                              if self.stride > 1 else "")
                hud = (f"[{self.idx+1}/{total}]  {labeled}/{to_label} labeled"
                       f"{stride_tag}  {mode_tag}  {view_tag}  {status}")
                cv2.putText(canvas, hud, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255) if court else (0, 200, 255), 1)

                cv2.imshow(self.window, canvas)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    self.idx = -1
                    break
                elif key == ord("b"):
                    previous = [i for i in sorted(self.anchors) if i < self.idx]
                    self.idx = previous[-1] if previous else 0
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
                elif key == ord("a") or key in (13, 32):
                    # Accepting a suggestion is the common case, so it shares
                    # Enter/Space with "save what is on screen". When there is
                    # no suggestion the old behaviour is unchanged.
                    if self.suggestion and not self.current_box:
                        self._accept_suggestion()
                    else:
                        self._advance()
                    break

        cv2.destroyAllWindows()
        label_assist.save_sidecar(self.labels_dir, self.provenance)

        labeled = sum(1 for im in self.images
                      if os.path.exists(self._label_path(im)))
        print(f"\nDone: {labeled}/{total} images labeled.")
        if self.accepted_this_run or self.drawn_this_run:
            print(f"  {self.accepted_this_run} accepted from suggestions, "
                  f"{self.drawn_this_run} done by hand.")
        if self.stride > 1:
            print(f"  Working at stride {self.stride}. Run with --fill to "
                  f"interpolate the {total - len(self.anchors)} frames in "
                  f"between.")

    def fill(self, max_gap):
        """Interpolate the frames between labelled anchors.

        Only touches frames nobody has labelled, and refuses any gap wider
        than `max_gap`. Every frame it writes is recorded in the sidecar as
        interpolated rather than observed.
        """
        points = {}
        for index, img_path in enumerate(self.images):
            path = self._label_path(img_path)
            if not os.path.exists(path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            self._img_shape = img.shape[:2]
            box = self._load_existing(img_path)
            points[index] = (None if box is None
                             else ((box[0] + box[2]) / 2,
                                   (box[1] + box[3]) / 2))

        filled = label_assist.interpolate(points, max_gap=max_gap)
        if not filled:
            print("Nothing to fill: no gaps small enough between two labels.")
            return

        half = (self.box_size or 30) // 2
        written = 0
        for index, (x, y) in sorted(filled.items()):
            img_path = self.images[index]
            img = cv2.imread(img_path)
            if img is None:
                continue
            self._img_shape = img.shape[:2]
            self._save_label(img_path,
                             (int(x - half), int(y - half),
                              int(x + half), int(y + half)))
            name = os.path.basename(img_path)
            if name not in self.provenance["filled"]:
                self.provenance["filled"].append(name)
            written += 1

        label_assist.save_sidecar(self.labels_dir, self.provenance)
        print(f"Filled {written} frames by interpolation (max gap {max_gap}).")
        print(f"These are inferred, not observed. They are listed under "
              f"'filled' in {label_assist.SIDECAR} so an evaluation set can be "
              "filtered back down to frames a person actually saw.")
        print("Measured on clip_01 at stride 3: 3.8 px median, 83% within "
              "12 px, but ~10% over 23 px out and ~8% place a shuttlecock on a "
              "frame that was actually empty. Good enough to train on; not "
              "good enough to measure with.")


def main():
    ap = argparse.ArgumentParser(description="Shuttlecock labeling tool")
    ap.add_argument("--images", type=str, default="datasets/images",
                    help="Directory of images to label")
    ap.add_argument("--labels", type=str, default="datasets/labels",
                    help="Directory for YOLO label files")
    ap.add_argument("--box-size", type=int, default=None,
                    help="Fixed square box size (click mode).")
    ap.add_argument("--no-suggest", dest="suggest", action="store_false",
                    default=True,
                    help="Do not pre-seed boxes with TrackNet's detections.")
    ap.add_argument("--model", type=str, default=None,
                    help="TrackNet weights to suggest from.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Label every Nth frame only; --fill does the rest.")
    ap.add_argument("--fill", action="store_true",
                    help="Interpolate unlabelled frames between labels "
                         "and exit, instead of opening the labeler.")
    ap.add_argument("--max-fill-gap", type=int,
                    default=label_assist.DEFAULT_MAX_FILL_GAP,
                    help="Widest gap between two labels that --fill will "
                         "interpolate across.")
    args = ap.parse_args()

    labeler = Labeler(args.images, args.labels, args.box_size,
                      suggest=args.suggest and not args.fill,
                      stride=args.stride, model_path=args.model)
    if not labeler.images:
        return
    if args.fill:
        labeler.fill(args.max_fill_gap)
    else:
        labeler.run()


if __name__ == "__main__":
    main()
