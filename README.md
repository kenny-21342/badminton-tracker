# Badminton Shuttlecock Trajectory Tracker

Tracks the path of the shuttlecock on your screen in real time while you watch
a badminton livestream (browser, media player, anything on screen). Draws a
color-coded trail of the rally and counts hits.

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

- **Colored trajectory trail** – a glowing polyline following the shuttlecock.
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
in *System Settings > Privacy & Security* and restart.

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
python3 main_screen.py --smooth 0.4             # lower = snappier, higher = smoother
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
| `ui_overlay.py` | **NEW:** ImGui-based settings overlay panel |
| `config_manager.py` | **NEW:** Settings persistence system |
| `session_recorder.py` | **NEW:** Trajectory recording and replay |
| `region_selector.py` | Interactive drag-to-select capture region |
| `extract_frames.py` | Sample frames from video/screen for labeling |
| `labeler.py` | Interactive labeling tool (drag boxes around shuttle) |
| `train.py` | Fine-tune YOLOv8 on your labeled frames |
| `download_tracknet.py` | Download pre-trained TrackNetV2 weights |
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

## Limitations

- **Classic CV**: very fast smashes can be missed between frames; bright court
  lines / white jerseys can create false positives. Threshold tuning is needed
  per broadcast.
- **YOLO**: requires one-time labeling effort (~1 hr for 200 frames). Accuracy
  depends on label quality and dataset diversity.
- **TrackNet**: at full 512×288 the pre-trained model runs at 4–6 FPS on Apple
  Silicon; use `--tracknet-input 256x144 --tracknet-fp16` for ~4x speed (some
  accuracy loss), or process recorded videos offline.
- **Hit counting** is heuristic and works best with a stable camera and a
  clear side-view of the court.
