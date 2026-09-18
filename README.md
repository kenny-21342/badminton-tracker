# Badminton Shuttlecock Trajectory Tracker

Tracks the path of the shuttlecock on your screen in real time while you watch
a badminton livestream (browser, media player, anything on screen). Draws a
marks the shuttlecock on screen and counts hits (press `p` for a
color-coded trail of the rally instead).

Supports three detection modes:
- **TrackNet AI** (`--tracknet`) — **Zero manual labeling required!** Deep-learning temporal heatmap model (TrackNetV2) pre-trained on professional badminton broadcast footage. Processes 3 consecutive frames and outputs heatmaps that locate fast-moving, blurred shuttlecocks.
- **Classic CV** (default) — motion + brightness heuristics, no setup needed.
- **YOLO AI** (`--yolo`) — single-frame object detector fine-tuned on custom labeled footage.

---

## Quick start (TrackNet AI mode — Zero-Labeling ⭐ Recommended)

```bash
cd badminton-tracker
pip3 install -r requirements.txt
python3 download_tracknet.py          # Downloads pre-trained TrackNetV2 weights (~136 MB) to models/
python3 main_screen.py --tracknet      # Runs with TrackNet temporal heatmap detector
```

`download_tracknet.py` fetches TrackNetV2 weights trained on badminton
broadcast footage from the open
[TrackNet Series PyTorch](https://github.com/AnInsomniacy/tracknet-series-pytorch)
project. Direct link:

> https://github.com/AnInsomniacy/tracknet-series-pytorch/releases/download/model-v2.0.1/tracknet_v2_epoch30.pt

If the download fails, open that link in your browser and save the file as
`models/TrackNet_best.pt`, then run `python3 main_screen.py --tracknet`.

Run `python3 benchmark.py` to measure these on your own machine. It feeds
synthetic frames straight to each detector, so the figures are upper bounds
with no capture or drawing in the way.

### TrackNet performance tips

Measured on an Apple M3 (10-core GPU), full pipeline including screen
capture and drawing:

| Engine | 512×288 | 384×216 | 256×144 |
|--------|---------|---------|---------|
| PyTorch (MPS) fp16 | ~11 FPS | ~18 FPS | ~41 FPS |
| **MLX fp16 (recommended)** | ~12 FPS | ~21 FPS | **~42 FPS** |

Install the optional MLX engine (2x faster at reduced resolutions):

```bash
pip3 install mlx
```

```bash
python3 main_screen.py --tracknet --tracknet-engine mlx \
    --tracknet-input 256x144 --tracknet-fp16
```

`256x144` + MLX runs at ~40 FPS. `384x216` is a good middle ground (~21 FPS).

**Full-resolution localization for free:** by default the heatmap is
upsampled back to the capture resolution before the shuttlecock position is
extracted (add `--no-tracknet-upsample` to disable). This means even at
`256x144` inference you get near full-resolution positioning precision, at
under 1 ms extra cost. The 512×288 ceiling on this GPU is ~12 FPS in any
framework (PyTorch or MLX); a CoreML export (`export_tracknet_coreml.py`) is
provided but needs a Python <= 3.13 interpreter because coremltools has no
Python 3.14 builds.

## Quick start (Classic CV mode)

```bash
cd badminton-tracker
pip3 install -r requirements.txt
python3 main_screen.py
```

A snapshot of your primary monitor appears. **Drag a rectangle around the
badminton court** in your livestream window, then press **ENTER**.

## Quick start (YOLO AI mode)

The AI mode requires a one-time training session (~30 min) but then
detects the shuttlecock with far higher accuracy. Four steps:

### Step 1 — Download the base model

Download `yolov8n.pt` (6 MB) in your browser:

> https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt

Save it in the project directory.

### Step 2 — Extract frames for labeling

```bash
# From a recorded match video:
python3 extract_frames.py --video match.mp4 --count 200

# Or capture frames from your live screen:
python3 extract_frames.py --screen --count 50 --interval 2
```

Frames are saved to `datasets/images/`. Aim for 100-300 frames covering
diverse shuttle positions (fast smashes, slow drops, near net, near baseline).

### Step 3 — Label the shuttlecock

Most frames do not need a box drawn — they need one confirmed. `labeler.py`
pre-seeds TrackNet's detection in amber; `a` (or SPACE) accepts it, and it is
not a label until you do. Measured over `clip_01` + `clip_02`:

| what the frame needs | frames | share |
|---|---|---|
| accept the suggestion (1 key) | 176 | 58.3% |
| confirm "no shuttle" (1 key) | 26 | 8.6% |
| correct a wrong box | 19 | 6.3% |
| reject a false positive | 2 | 0.7% |
| draw from scratch | 79 | 26.2% |

**67% of frames become one keypress**, and accepted suggestions sit 3.6 px from
the hand labels. Suggestions need consecutive frames — TrackNet finds the
shuttle partly by where it moved — so on a directory of sampled frames the tool
says so and turns them off rather than suggesting noise. `--no-suggest`
disables them.

For **training** sets you can go further and label only every Nth frame:

```bash
python3 labeler.py --images clip_03 --labels labels_03 --stride 3
python3 labeler.py --images clip_03 --labels labels_03 --fill
```

At stride 3 that is a third of the work, and filling reproduces held-out labels
to 3.8 px median with 83% inside 12 px. **Do not do this for an evaluation
set**: ~10% of filled frames land over 23 px out, and ~8% place a shuttlecock
on a frame that was actually empty, which would flatter recall in the very
measurement the set exists for. Filled frames are recorded under `filled` in
`_assist.json` so a set can always be cut back to observed frames.

Two approaches that were tried and **rejected on measurement**: interpolating
across the detector's own gaps (only ~50% within 12 px — it misses exactly the
fast, blurred frames where a straight line is worst), and adding anchors where
the detected path bends most (6 points more labelling, no accuracy gain).

```bash
python3 labeler.py
```

A window opens showing each frame. **Drag a box around the shuttlecock**
and press **ENTER**. If the shuttle isn't visible in a frame, press **N**
to skip. Controls:

| Key | Action |
|-----|--------|
| Mouse drag | Draw box around shuttle |
| `ENTER` / `SPACE` | Save and next frame |
| `N` | No shuttle visible — skip |
| `B` | Go back to previous frame |
| `D` | Delete current box and re-draw |
| `Q` | Quit (progress saved) |

**Faster labeling** — if the shuttle is small, use a fixed-size click box:
```bash
python3 labeler.py --box-size 30   # single click places a 30px box
```

Labels are saved to `datasets/labels/` in YOLO format.

### Step 4 — Train

`train.py` splits train/val by *group*, not by frame. Neighbouring frames of a
rally are near-identical, so splitting them at random puts a frame in train and
the frame 40 ms later in val, and the validation score stops measuring
generalisation. Contiguous clips are therefore kept whole; sampled frames are
far enough apart to split individually. Which a directory is gets decided from
the pixels, not the filenames — `extract_frames.py` numbers sampled frames
`0000, 0001, 0002` even when they are minutes apart — and the measurement is
printed so you can check it:

```
  images:  1000 frames (neighbour diff 4.84 — sampled, split individually)
  clip_03:  151 frames (neighbour diff 0.85 — contiguous, kept together)
```

Fold extra clips in with `--add`, repeatable:

```bash
python3 train.py \
    --add datasets/eval_ld2011/clip_03:datasets/eval_ld2011/labels_03 \
    --add datasets/eval_ld2011/clip_04:datasets/eval_ld2011/labels_04
```

`datasets/yolo/` is rebuilt from scratch each run. It used to be written
incrementally and skip files already present, so a frame that landed in val on
one run and train on the next stayed in both — training on part of the
validation set, silently. Adding `clip_01` or `clip_02` also prints a warning:
those are what `evaluate.py --clips` measures against.

```bash
python3 train.py
```

This fine-tunes YOLOv8-nano on your labels (80 epochs by default).
The trained model is saved to `runs/detect/train/weights/best.pt`.

### Step 5 — Run with AI detection

```bash
python3 main_screen.py --yolo
```

The tracker automatically loads your trained model. If no model is found,
it falls back to classic CV mode. You can also specify a model path:

```bash
python3 main_screen.py --yolo-model path/to/best.pt
python3 main_screen.py --yolo --yolo-conf 0.15   # lower = more sensitive
```

---

## What you see on screen (the result)

- **Shuttle marker** – a ring on the shuttlecock; solid when detected, hollow
  when the position was bridged across a gap. This is the default.
- **Colored trajectory trail** – a glowing polyline following the shuttlecock,
  on `p`.
  **Green = slow** (drop shots / net play), **yellow = medium**, **red = fast**
  (smashes / clears). Older points fade out; the newest position is a red dot.
- **"HIT!" flash** – when a direction reversal or sharp deceleration is
  detected (a racket contact), the screen flashes briefly and a counter ticks.
- **Top HUD bar** – `Rally`, `Hits`, `Speed`, `FPS`, and detection diagnostics.
- **Trajectory Map (mini-map, top-right)** – a scaled-down bird's-eye replay
  of the full rally path.
- **Bottom bar** – the list of live controls.

## Installation

```bash
cd badminton-tracker
pip3 install -r requirements.txt
```

On first run macOS may prompt for **Screen Recording** permission — grant it
in *System Settings > Privacy & Security*, then quit and reopen the app. The
grant only applies to freshly launched processes.

macOS does not report a refusal as an error: a denied capture succeeds and
returns a blank frame, so without a check the tracker runs against a black
screen and looks like a detector that cannot find anything. The app now tests
one frame at startup and stops with an explanation instead.

### Building the macOS app

```bash
pip3 install pyinstaller
python3 -m PyInstaller --noconfirm BadmintonTracker.spec
open "dist/Badminton Tracker.app"
```

The bundle is ~940 MB, most of it PyTorch, and ships the TrackNet weights so it
works on first launch with no download. One known limitation: MLX fails to
initialise inside the bundle, so the packaged app runs TrackNet on Torch/MPS
rather than MLX. Measured cost at 512x288 is 4.8 fps against 5.4 — about 12%.
Running from source still uses MLX.

### Rendering a demo clip

```bash
python3 make_demo.py --input match.mp4 --start 773 --duration 12 \
    --output demo.mp4
```

Runs the same detector, tracker and visualiser the live app uses, but reads a
video file and writes an annotated mp4 — no screen capture, no recorder, and
the result is reproducible.

Finding a rally to point it at is `extract_frames.py`'s job — `find_rallies()`
scores every frame against the empty court and returns the unbroken wide-court
stretches. On the 2011 final it finds 466 shots, 85 of them over six seconds.
The clip at 773 s renders 147 detections across 300 frames, 12 hits and 3
rallies.

Rendering is not real time: about 13 s of compute per second of 1280×720
footage, since it decodes the file and runs TrackNet on every frame.

The rendered clip is not committed, for the same reason `datasets/` and the
source `.mp4` are not — broadcast footage is not ours to redistribute.

## Usage reference

### Capture modes

```bash
python3 main_screen.py                          # interactive region picker
python3 main_screen.py --monitor 1              # whole primary monitor
python3 main_screen.py --region "100,80,1280,720"  # explicit region
```

### Performance vs. accuracy

```bash
python3 main_screen.py --scale 0.4              # classic CV: smaller = faster
python3 main_screen.py --smooth 0.4             # ceiling on prediction blending (see below)
python3 main_screen.py --display-scale 0.75     # shrink preview window
python3 main_screen.py --tracknet --tracknet-input 256x144 --tracknet-fp16  # TrackNet speed mode
```

## Live controls (in the tracker window)

| Key | Action | Mode |
|-----|--------|------|
| `SPACE` | **Pause / resume tracking** (for replays, adverts, between games) | all |
| `TAB` | **Toggle settings overlay panel** (needs imgui — see below) | all |
| `?` | **Toggle help screen** | all |
| `c` | Clear the trajectory (new rally) | all |
| `r` | Reset the rally counter to 0 | all |
| `m` | Toggle the mini-map | all |
| `p` | Trail ↔ point overlay | all |
| `h` | Toggle the HUD bars | all |
| `s` | Toggle smoothing level (0.2 / 0.6) | all |
| `v` | **Start / stop recording the session** (writes JSON to `sessions/`) | all |
| `d` | Toggle debug masks view | classic |
| `+` / `-` | Brightness threshold up / down | classic |
| `[` / `]` | Motion threshold / detector confidence (YOLO & TrackNet) | all |
| `w` / `e` | Saturation down / up | classic |
| `,` / `.` | Smoothing down / up (fine) | all |
| `q` | Quit | all |

The settings panel needs `imgui`, which publishes no wheel for Python 3.14 and
fails to build from source there. Without it the tracker runs normally, says so
at startup, and prints the messages the panel would have shown; every shortcut
above still works. On Python 3.13 or older, `pip3 install "imgui[opencv]"`
enables the panel.

Pausing stops detection while the window stays live, closes the current rally
and clears the trail, so a replay is not tracked and its movement is not
counted as play.

Pressing `v` records the whole session — per-frame position, velocity and
speed, plus hit counts and the length of every rally — to a timestamped JSON
file under `sessions/`. Recording also stops and saves on quit, including on
Ctrl-C. `SessionRecorder.export_to_csv` converts one to CSV.

### New: Settings Overlay Panel (Press TAB)

The overlay provides real-time control over all tracker parameters without needing to remember keyboard shortcuts:

**Detection Tab**
- Switch detector modes on-the-fly
- Adjust confidence thresholds with sliders
- View live detector status (🟢 Locked | 🟡 Coasting | 🔴 Lost)

**Tracking Tab**
- Fine-tune smoothing (0.0-0.95)
- Adjust hit detection sensitivity
- Configure rally cooldown period

**Display Tab**
- Toggle HUD, minimap, and debug views
- Adjust trail colors and opacity
- Switch between dark/light themes

**Session Tab**
- Record trajectory data to JSON
- Replay saved sessions
- Export trajectory data for analysis

All settings are automatically saved and restored on next launch.

### Classic CV tuning tips

Press **`d`** to open the debug masks window (frame | white | motion | AND).

- If the trail is **jittery or jumps to players**: raise brightness with `+`,
  or raise motion with `]`.
- If the shuttle **isn't detected at all**: lower brightness with `-`,
  or lower motion with `[`.
- If the trail **lags behind**: lower smoothing with `,` or `s`.
- Use the **region picker** to capture just the court — it dramatically
  improves detection and frame rate.

### YOLO tuning tips

- If the shuttle **isn't detected**: lower confidence with `[`.
- If **false positives appear**: raise confidence with `]`.
- Label more frames and retrain — accuracy improves with more data.

---

## Files

| File | Purpose |
|------|---------|
| `main_screen.py` | Screen-capture app — the main entry point |
| `detector.py` | Classic CV shuttlecock detection |
| `yolo_detector.py` | YOLO + hybrid detector (AI mode) |
| `tracknet_detector.py` | TrackNet temporal heatmap detector (PyTorch) |
| `mlx_tracknet_detector.py` | MLX-accelerated TrackNet for Apple Silicon |
| `tracker.py` | Trajectory smoothing, gap-filling, hit detection |
| `visualizer.py` | Trail, HUD, mini-map, and theme rendering |
| `ui_overlay.py` | ImGui settings overlay panel (optional — see controls) |
| `config_manager.py` | Settings persistence |
| `session_recorder.py` | Session recording, replay and CSV export |
| `region_selector.py` | Interactive drag-to-select capture region |
| `extract_frames.py` | Sample frames from video/screen for labeling |
| `labeler.py` | Interactive labeling tool (drag boxes around shuttle) |
| `label_assist.py` | Detector suggestions and sparse-label interpolation |
| `train.py` | Fine-tune YOLOv8 on your labeled frames (group-aware split) |
| `download_tracknet.py` | Download pre-trained TrackNetV2 weights |
| `export_tracknet_coreml.py` | Export TrackNet to CoreML (needs Python <= 3.13) |
| `evaluate.py` | Measure detector accuracy against labelled frames |
| `benchmark.py` | Measure detector speed on synthetic frames |
| `frame_classifier.py` | Court-view classifier used when sampling frames |
| `make_demo.py` | Render an annotated demo clip from a video file, headlessly |
| `screen_permission.py` | Detects a denied Screen Recording permission before capture |
| `resources.py` | Resolves bundled data files when running from a .app |
| `BadmintonTracker.spec` | PyInstaller recipe for the double-clickable macOS app |
| `main.py` | Legacy entry point for video files / webcams |

## How it works

### Classic CV mode

1. **Capture** — `mss` grabs the screen region at 30-60 fps.
2. **Detect** (`detector.py`) — the frame is downscaled; HSV "white" extraction
   + background subtraction produce a candidate mask. A prediction-gated
   tracker searches only near the predicted next position, rejecting jerseys,
   court lines, and scorebugs. Velocity-based cold-start picks the fastest
   bright blob as the shuttle.
3. **Track** (`tracker.py`) — detections are smoothed via velocity prediction;
   missing frames are bridged by coasting. Hit detection fires on direction
   reversal or large deceleration.
4. **Visualize** (`visualizer.py`) — speed-colored trail, marker, HIT flash,
   HUD, mini-map.

### YOLO AI mode

1. **Capture** — same screen grab.
2. **Detect** (`yolo_detector.py`) — a fine-tuned YOLOv8-nano model runs
   inference on each frame, returning the shuttle bounding box with a
   confidence score. No brightness/motion thresholds needed — the model
   learned shuttle appearance directly.
3. **Hybrid fallback** — if YOLO finds nothing (extreme motion blur), the
   classic CV detector fills the gap automatically.
4. **Track + Visualize** — identical to classic mode.

### TrackNet AI mode

1. **Capture** — same screen grab.
2. **Detect** (`tracknet_detector.py`) — the last 3 captured frames are
   concatenated (9 channels) and passed through a TrackNetV2 U-Net that
   outputs a per-frame heatmap. The largest blob above the confidence
   threshold becomes the shuttlecock position.
3. **Hybrid fallback** — if TrackNet's heatmap confidence is low, the classic
   CV detector fills the gap automatically.
4. **Track + Visualize** — identical to classic mode.

### Overlay style: trail or point

Point is the default; `p` toggles, `--style trail` starts on the trail. The
trail draws the whole tracked path; point draws only where the shuttle is now.

On broadcast footage the trail is a long streak across the frame, much of it
extrapolated through gaps, sitting on top of the thing it points at. The point
is far more legible. What makes it workable is that it still uses the tracker's
gap filling: measured on `clip_01`, a marker drawn only on real TrackNet
detections is present for **61%** of in-play frames, while gap filling carries
it to **86%**.

Dropouts are mostly invisible — 10 of 17 last one or two frames (40–80 ms) —
but the tail is real, up to 13 frames (520 ms). So a bridged position is drawn
**hollow** and a real detection **solid**: the overlay never asserts a sighting
it does not have.

This changes what you see, not what is counted. Hit detection and rally stats
read the trajectory, so recall still matters for the numbers even when the
display no longer looks gappy.

## Limitations

- **Classic CV**: very fast smashes can be missed between frames; bright court
  lines / white jerseys can create false positives. Threshold tuning is needed
  per broadcast.
- **YOLO**: requires one-time labeling effort (~1 hr for 200 frames). Accuracy
  depends on label quality and dataset diversity.
- **TrackNet**: at full 512×288 the pre-trained model runs at about 13 FPS on
  an M3 (fp16, MPS), or 41 at `--tracknet-input 256x144 --tracknet-fp16`. Run
  `python3 benchmark.py` for your own machine. An earlier version of this
  section said 4–6 FPS, which contradicted the table above it and was wrong.
- **Hit counting** is heuristic and works best with a stable camera and a
  clear side-view of the court. Press `SPACE` during replays and adverts:
  movement on screen is counted as play otherwise. Its 25 px/frame threshold
  has never been checked against hit-level ground truth — see the smoothing
  section under "What has actually been measured".
- **The packaged .app runs TrackNet on Torch/MPS, not MLX.** MLX fails to
  initialise inside the PyInstaller bundle. Costs about 12% (4.8 fps against
  5.4 at 512×288). Running from source is unaffected.

## What has actually been measured

Worth being straight about, because most of the claims above are impressions
rather than measurements.

### The sampled set (`datasets/`, 1000 frames)

Only YOLO can be judged on it, and only on the 200 frames it did not train on:
**79.5% recall, 92.5% precision, 2.0 px median error**. Its PCK is identical
at 8, 25 and 50 px, which is what memorising a training set looks like rather
than generalising.

Classic CV and TrackNet cannot be judged on that set at all. Both need
consecutive frames — classic CV compares against the previous one, TrackNet
reads three at a time — and the set is 1000 sampled frames in 388 runs with a
median run length of one. Their numbers there measure the dataset, not them.

### The contiguous set (`datasets/eval_ld2011/`, 302 labelled frames)

`python3 evaluate.py --clips` — 274 frames with a shuttlecock, 28 without,
across `clip_01` and `clip_02` (151 consecutive frames each). Tolerance 25 px.

| detector | recall | precision | FP rate | median err | PCK@8 | fps |
|---|---|---|---|---|---|---|
| classic CV | 0.4% | 0.3% | 100% | 714 px | 0.0% | 357 |
| **TrackNet** | **70.1%** | **97.4%** | **7.1%** | **4.0 px** | **59.3%** | **9.1** |
| YOLO\* | 37.2% | 34.1% | 100% | 152 px | 35.0% | 56 |

\* YOLO was fine-tuned on frames sampled from this same match. None of these
files were in its training set, but frames a fraction of a second away were,
so its row is still not clean held-out data — and it is the worst of the two
learned detectors regardless.

**TrackNet is the default**, and this is the evidence for it. It is the only
detector that is right more often than not, its 97.4% precision means a
detection can be trusted rather than filtered, and 4 px median error is inside
the shuttlecock's own 16 px box. It pays for that at 9 fps, roughly a sixth of
YOLO's rate — use `--tracknet-input 256x144 --tracknet-fp16` when live speed
matters more than accuracy.

Two honest caveats:

- **This is broadcast footage, which is not what classic CV is for.** It
  tracks a white, moving blob, and a broadcast frame is full of white-clad
  players, line judges and crowd. On a fixed camera pointed at one court —
  the app's actual use case — it does considerably better than 0.4%. Its row
  here says "do not use this on television", not "this code is broken".
- **Two clips is not enough.** The measurement wants 5–10, and
  `clip_03`–`clip_08` are extracted but unlabelled (`evaluate.py --clips`
  names them as skipped on every run). Label them with `labeler.py` and rerun;
  the numbers above should be treated as provisional until then.

### Smoothing (`--smooth`)

Same clips, same detector, varying only the smoothing blend. Error is against
the labelled positions on frames that had a real detection; "gap err" is on
frames the tracker filled in by prediction; "jerk" is the mean second
difference along the drawn trail, a proxy for how steady it looks.

| setting | median err | mean err | mean jerk | gap err |
|---|---|---|---|---|
| raw detections | 4.00 px | 5.33 px | — | — |
| fixed 0.1 | 4.53 px | 6.85 px | 19.8 | 50.6 px |
| fixed 0.5 *(old default)* | 9.01 px | 21.56 px | 14.7 | 53.5 px |
| fixed 0.9 | 161.00 px | 206.18 px | 23.7 | 288.1 px |
| **adaptive (any ceiling)** | **~4.00 px** | **~5.4 px** | 24.4 | **30.9 px** |

A fixed blend cost accuracy at every setting, and the old default of 0.5 more
than doubled the median error. The cause is lag: a one-step linear-plus-gravity
prediction cannot keep up with a shuttlecock, so blending towards it drags the
drawn point behind the real one. Past 0.6 the prediction diverges and the trail
is not even steady any more.

The steadiness a fixed blend appeared to buy was mostly that same lag rather
than noise suppression, and there is no good trade on the curve: loosening the
adaptive tolerance to buy smoothness back moves mean jerk only from 24.5 to
22.0 while median error goes from 4.0 to 5.4.

So smoothing is now **adaptive** and `--smooth` sets a ceiling rather than a
fixed blend. It is scaled down when the detection disagrees with the prediction
(a hit, a direction change) and when the track is too young for its velocity
estimate to mean anything — leaving it to act only where it is close to free.
`TrajectoryTracker(adaptive_smoothing=False)` restores the old behaviour.

One thing this does **not** claim: that hit counting improved. Hit detection's
25 px/frame threshold was calibrated while trajectories were lagged, and the
old smoother overshot direction changes enough to inflate the measured velocity
change. On ground truth, real vertical reversals span 5.5–94 px/frame with a
median of 19.4, so the threshold sits inside the distribution with no clean
separation to move it to. Choosing a better value needs hit-level ground truth,
which the eval set does not have.
