"""
Badminton Trajectory Tracker - Screen Capture Mode.

Captures a region of your screen (e.g. a browser livestream) in real time,
detects the shuttlecock, and marks it on screen (press p for the
full trajectory trail instead).

Quick start:
    python3 main_screen.py                 # region picker; TrackNet by default
    python3 main_screen.py --cv            # force the classic CV detector
    python3 main_screen.py --monitor 1     # whole primary monitor
    python3 main_screen.py --region "100,80,1280,720"
    python3 main_screen.py --yolo          # use trained YOLO model
    python3 main_screen.py --yolo-model models/my_model.pt

Controls (in the tracker window):
    c   clear trajectory            +/-  brightness threshold up/down (classic)
    r   reset rally counter         [, ] motion threshold / YOLO confidence
    m   toggle minimap              , /. smoothing down/up
    p   trail <-> point overlay
    s   toggle smoothing            d   toggle debug masks view (classic)
    h   toggle HUD                  w/e saturation down/up (classic)
    q   quit

When using --yolo, the classic CV controls have no effect (YOLO doesn't
need brightness/motion tuning). Use [ and ] to adjust YOLO confidence.
"""
import os
import sys
import cv2
import numpy as np
import time
import argparse
from mss import mss

from tracker import TrajectoryTracker
from visualizer import TrajectoryVisualizer
from region_selector import select_region
from ui_overlay import OverlayUI
from config_manager import ConfigManager
from session_recorder import SessionRecorder
from screen_permission import check_capture_permitted, report_problem


def resolution_arg(value):
    """Parse a WxH resolution argument (sides divisible by 8)."""
    try:
        w, h = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid resolution {value!r}: expected WxH, e.g. 512x288")
    if w < 8 or h < 8 or w % 8 != 0 or h % 8 != 0:
        raise argparse.ArgumentTypeError(
            f"resolution {value!r}: sides must be divisible by 8")
    return w, h


def tracknet_weights_available():
    """Whether TrackNet can load without a download."""
    from tracknet_detector import DEFAULT_MODEL_PATHS

    return any(os.path.exists(p) for p in DEFAULT_MODEL_PATHS)


def create_detector(args):
    """Create the appropriate detector based on flags.

    With no detector flag at all, TrackNet is the default when its weights are
    on disk. That is a measurement, not a preference: on the contiguous clips
    in datasets/eval_ld2011 it gets 70.1% recall at 97.4% precision with 4 px
    median error, against 37.2%/34.1% for YOLO and effectively nothing for
    classic CV (see "What has actually been measured" in the README). Classic
    CV remains the fallback when the weights are missing, and `--cv` forces it
    — it is six times faster on broadcast-sized frames and it is the right
    choice on a fixed single-court camera, which is not what the clips are.
    """
    explicit = (args.cv or args.tracknet or args.tracknet_model
                or args.yolo or args.yolo_model)
    if not explicit and tracknet_weights_available():
        print("No detector specified - using TrackNet (the measured default). "
              "Pass --cv for the classic detector.")
        args.tracknet = True

    if args.tracknet or args.tracknet_model:
        try:
            from tracknet_detector import TrackNetHybridDetector
            iw, ih = args.tracknet_input
            detector = TrackNetHybridDetector(
                model_path=args.tracknet_model,
                confidence=args.tracknet_conf,
                classic_scale=args.scale,
                enable_fallback=True,
                device=args.tracknet_device,
                input_width=iw,
                input_height=ih,
                fp16=args.tracknet_fp16,
                engine=args.tracknet_engine,
                upsample_heatmap=args.tracknet_upsample,
            )
            if detector.tracknet is None:
                # TrackNet failed to load; use the classic detector directly so
                # the main loop receives plain points, not hybrid tuples.
                print("TrackNet unavailable - using classic CV detector.")
                return detector.classic, "classic"
            return detector, "tracknet"
        except Exception as e:
            print(f"Warning: could not load TrackNet ({e})")
            print("Falling back to classic CV detector.\n")
            from detector import ShuttlecockDetector
            return ShuttlecockDetector(scale=args.scale), "classic"
    elif args.yolo or args.yolo_model:
        try:
            from yolo_detector import HybridDetector
            detector = HybridDetector(
                yolo_path=args.yolo_model,
                classic_scale=args.scale,
                confidence=args.yolo_conf,
                enable_fallback=True,
                imgsz=args.yolo_imgsz,
                device=args.yolo_device,
            )
            if detector.yolo is None:
                # YOLO model missing; use the classic detector directly so the
                # main loop receives plain points, not hybrid tuples.
                print("YOLO model unavailable - using classic CV detector.")
                return detector.classic, "classic"
            return detector, "hybrid"
        except Exception as e:
            print(f"Warning: could not load YOLO ({e})")
            print("Falling back to classic CV detector.\n")
            from detector import ShuttlecockDetector
            return ShuttlecockDetector(scale=args.scale), "classic"
    else:
        from detector import ShuttlecockDetector
        return ShuttlecockDetector(scale=args.scale), "classic"


def main():
    ap = argparse.ArgumentParser(
        description="Badminton Trajectory Tracker - Screen Capture")
    ap.add_argument("--monitor", type=int, default=None,
                    help="Monitor index (1=primary, 0=all).")
    ap.add_argument("--region", type=str, default=None,
                    help='Region as "x,y,width,height" in screen pixels.')
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Classic detector processing scale (0.25-1.0).")
    ap.add_argument("--display-scale", type=float, default=None,
                    help="Scale the preview window (auto-fit by default).")
    ap.add_argument("--style", choices=("trail", "point"), default="point",
                    help="Overlay style: a marker on the shuttle (default), or "
                         "the full trail. Toggle live with p.")
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="Tracker smoothing 0-1 (0=raw, 1=prediction).")
    ap.add_argument("--cv", action="store_true",
                    help="Force the classic CV detector. Without any detector "
                         "flag, TrackNet is used when its weights are present.")
    ap.add_argument("--tracknet", action="store_true",
                    help="Use TrackNet temporal heatmap detector (pre-trained PyTorch model).")
    ap.add_argument("--tracknet-model", type=str, default=None,
                    help="Path to TrackNet .pt model weights file.")
    ap.add_argument("--tracknet-conf", type=float, default=0.5,
                    help="TrackNet heatmap confidence threshold (0-1).")
    ap.add_argument("--tracknet-device", type=str, default=None,
                    help="TrackNet device: cpu, mps, cuda.")
    ap.add_argument("--tracknet-input", type=resolution_arg, default=(512, 288),
                    metavar="WxH",
                    help="TrackNet input resolution (sides divisible by 8; "
                         "e.g. 512x288 default, 256x144 = ~4x faster).")
    ap.add_argument("--tracknet-fp16", action="store_true",
                    help="Run TrackNet in float16 (faster on Apple "
                         "Silicon / GPU).")
    ap.add_argument("--tracknet-engine", type=str, default="auto",
                    choices=["auto", "torch", "mlx"],
                    help="TrackNet inference engine: auto (default), "
                         "torch (PyTorch/MPS), or mlx (2x faster on Apple "
                         "Silicon; requires 'pip install mlx').")
    ap.add_argument("--no-tracknet-upsample", dest="tracknet_upsample",
                    action="store_false", default=True,
                    help="Disable heatmap upsampling to full resolution "
                         "(slightly faster, coarser localization).")
    ap.add_argument("--yolo", action="store_true",
                    help="Use YOLO detector (requires trained model).")
    ap.add_argument("--yolo-model", type=str, default=None,
                    help="Path to YOLO .pt model file.")
    ap.add_argument("--yolo-conf", type=float, default=0.25,
                    help="YOLO confidence threshold (0-1).")
    ap.add_argument("--yolo-imgsz", type=int, default=640,
                    help="YOLO inference image size (lower=faster).")
    ap.add_argument("--yolo-device", type=str, default=None,
                    help="YOLO device: cpu, mps, or auto.")
    args = ap.parse_args()

    # --- Determine capture region -----------------------------------------
    sct = mss()

    # A denied capture is not an error on macOS, just a blank frame. Check
    # before the region picker, which would otherwise ask the user to drag a
    # selection over a black rectangle.
    permitted, message = check_capture_permitted(
        sct, sct.monitors[args.monitor if args.monitor is not None else 1])
    if not permitted:
        report_problem(message)
        return 1

    if args.region:
        x, y, w, h = map(int, args.region.split(","))
        monitor = {"left": x, "top": y, "width": w, "height": h}
        print(f"Capturing region: {w}x{h} at ({x},{y})")
    elif args.monitor is not None:
        monitor = sct.monitors[args.monitor]
        print(f"Capturing monitor {args.monitor}: "
              f"{monitor['width']}x{monitor['height']}")
    else:
        monitor = select_region(monitor_index=1)

    w, h = monitor["width"], monitor["height"]
    if w < 2 or h < 2:
        print("Error: invalid capture region.")
        return

    # --- Create detector --------------------------------------------------
    detector, mode = create_detector(args)
    tracker = TrajectoryTracker(max_trajectory_length=15,
                                hit_detection_threshold=25,
                                smoothing=args.smooth)
    visualizer = TrajectoryVisualizer()

    # --- UI Overlay and Config Management ---------------------------------
    config = ConfigManager()
    ui = OverlayUI()
    recorder = SessionRecorder()

    # Load saved settings
    tracker.smoothing = config.get("smoothing", args.smooth)
    visualizer.show_minimap = config.get("show_minimap", True)
    visualizer.style = config.get("style", args.style)
    visualizer.set_theme(config.get("theme", "dark"))
    visualizer.trail_opacity = config.get("trail_opacity", 0.8)

    # Classic detector returns downscaled coords; YOLO returns full-frame.
    # HybridDetector wraps classic internally; ShuttlecockDetector has .scale.
    if mode == "classic":
        if hasattr(detector, "classic"):
            det_scale = detector.classic.scale  # HybridDetector fallback
        else:
            det_scale = detector.scale           # pure ShuttlecockDetector
    else:
        det_scale = 1.0  # hybrid with YOLO: full-frame coords
    inv_scale = 1.0 / det_scale

    # Tracker frame size must match the coordinate space the detector emits.
    # Classic: downscaled coords. YOLO: full-frame coords.
    tracker.set_frame_size(int(w * det_scale), int(h * det_scale))

    # --- Preview scale ----------------------------------------------------
    disp_scale = args.display_scale
    if disp_scale is None:
        disp_scale = min(1.0, 1280.0 / max(w, 1))

    print("\nBadminton Trajectory Tracker")
    print("=" * 55)
    print(f"Capture: {w}x{h}  |  detector: {mode}  "
          f"|  preview scale: {disp_scale:.2f}")
    print("Controls: SPACE=pause  v=record session  c=clear  r=reset rally  "
          "m=minimap  s=smooth  +/-=bright  [/]=motion/conf  q=quit")
    print("=" * 55)

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0
    show_hud = True
    show_minimap = True
    show_debug = False
    paused = False

    try:
        while True:
            shot = sct.grab(monitor)
            frame = np.array(shot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # --- Detect ---------------------------------------------------
            if paused:
                # Keep capturing and drawing so the window stays live, but do
                # not detect: nothing on screen during a replay or an advert
                # should reach the trajectory or the hit count.
                det = det_full = None
                det_source = "paused"
            elif mode == "classic":
                det = detector.detect(frame)
                det_full = None
                if det is not None:
                    det_full = (int(det[0] * inv_scale),
                                int(det[1] * inv_scale))
                det_source = "cv"
            else:
                det_full, det_source = detector.detect(frame)
                det = det_full  # tracker works in full-frame space

            if not paused:
                tracker.add_point(det, frame=frame)

            # Map trajectory for drawing.
            if mode == "classic":
                traj = [(int(p[0] * inv_scale), int(p[1] * inv_scale))
                        for p in tracker.get_trajectory()]
                vels = [(v[0] * inv_scale, v[1] * inv_scale)
                        for v in tracker.get_velocities()]
            else:
                traj = tracker.get_trajectory()
                vels = tracker.get_velocities()

            # --- Draw -----------------------------------------------------
            # The tracker knows whether this position was detected or bridged;
            # the marker says so rather than implying a sighting.
            current_predicted = bool(tracker.point_is_predicted
                                     and tracker.point_is_predicted[-1])
            frame = visualizer.draw(frame, traj, vels, det_full, 0,
                                    current_is_predicted=current_predicted)
            # --- YOLO debug: draw bounding box + confidence ----------------
            if mode != "classic" and hasattr(detector, "yolo") and detector.yolo:
                yd = detector.yolo
                if hasattr(yd, "last_box") and yd.last_box:
                    bx1, by1, bx2, by2 = yd.last_box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{yd.last_conf:.2f}", (bx1, by1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Trail + HIT alert disabled — just show the detection marker.
            if show_minimap:
                frame = visualizer.draw_minimap(frame, traj, vels)
            if show_hud:
                extra = _build_hud(detector, tracker, mode, det_source, fps)
                frame = visualizer.draw_info(
                    frame, len(traj), fps, rally=tracker.rally_count,
                    hits=tracker.hit_count, speed=tracker.current_speed(),
                    extra=extra, rallies=tracker.rallies_completed,
                    paused=paused)

            # --- Session recording ---------------------------------------
            # SessionRecorder was constructed and handed to the overlay, but
            # nothing ever called start_recording, record_frame or
            # stop_recording — the class was complete and never driven, so no
            # session was ever written. Toggled with V.
            if recorder.is_recording:
                recorder.record_frame(
                    traj[-1] if traj else None,
                    tracker.last_velocity,
                    tracker.current_speed(),
                    tracker.hit_count,
                    tracker.rally_count,
                )

            # --- UI Overlay (ImGui panel) --------------------------------
            frame = ui.draw(frame, detector, tracker, visualizer, recorder, mode)

            # --- Debug masks window (classic only) -----------------------
            if show_debug and mode == "classic":
                dbg = detector.debug_frame(frame)
                cv2.imshow("Debug (frame | white | motion | AND)", dbg)

            # --- Preview scaling -----------------------------------------
            if disp_scale < 1.0:
                out = cv2.resize(frame, None, fx=disp_scale, fy=disp_scale,
                                 interpolation=cv2.INTER_AREA)
            else:
                out = frame
            cv2.imshow("Badminton Tracker (screen)", out)

            # --- FPS -----------------------------------------------------
            fps_n += 1
            if fps_n >= 20:
                fps = fps_n / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_n = 0
                # Hit detection compares pixels-per-frame velocities, and this
                # pipeline runs anywhere from ~2 fps to ~170 depending on the
                # detector. Telling the tracker the real rate keeps a hit the
                # same physical event instead of a function of the frame rate.
                if fps > 0:
                    tracker.observed_fps = fps

            # --- Keyboard ------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            print(f"[key] {key} ({chr(key) if 32 <= key < 127 else '?'})")

            if key == ord("q"):
                break
            elif key == ord("\t") or key == 9:  # TAB
                ui.show_panel = not ui.show_panel
            elif key == ord("?"):
                ui.show_help = not ui.show_help
            elif key == ord("c"):
                tracker.clear_trajectory()
                detector.reset()
                ui.toast("Trajectory cleared")
                print("[cleared trajectory]")
            elif key == ord("r"):
                tracker.reset_rally()
                ui.toast("Rally reset")
                print("[rally reset]")
            elif key == ord("m"):
                visualizer.show_minimap = not visualizer.show_minimap
                ui.toast(f"Minimap {'ON' if visualizer.show_minimap else 'OFF'}")
            elif key == ord("p"):
                visualizer.style = ("point" if visualizer.style == "trail"
                                    else "trail")
                ui.toast(f"Style: {visualizer.style}")
            elif key == ord("h"):
                show_hud = not show_hud
                ui.toast(f"HUD {'ON' if show_hud else 'OFF'}")
            elif key == ord("s"):
                tracker.smoothing = 0.2 if tracker.smoothing > 0.4 else 0.6
                ui.toast(f"Smoothing: {tracker.smoothing:.2f}")
                print(f"[smoothing = {tracker.smoothing}]")
            elif key == ord(" "):
                # Pause tracking without closing the window — for replays,
                # adverts and the gaps between games, where whatever is on
                # screen is not a rally and would otherwise be tracked and
                # counted as one.
                paused = not paused
                if paused:
                    # Whatever follows is not this rally continuing, and the
                    # old trail would smear into play when it resumes.
                    tracker.end_rally()
                    tracker.clear_trajectory()
                ui.toast("Paused" if paused else "Resumed")
            elif key == ord("v"):
                if recorder.is_recording:
                    saved = recorder.stop_recording(tracker)
                    ui.toast(f"Session saved: {saved.name}" if saved
                             else "Session save failed")
                    if saved:
                        print(f"[tracker] session written to {saved}", flush=True)
                else:
                    recorder.start_recording(mode)
                    ui.toast("Recording session…")
            elif key == ord("d"):
                show_debug = not show_debug
                if mode == "classic":
                    detector.debug = show_debug
            elif key == ord(","):
                tracker.smoothing = max(0.0, tracker.smoothing - 0.05)
                ui.toast(f"Smoothing: {tracker.smoothing:.2f}")
            elif key == ord("."):
                tracker.smoothing = min(0.95, tracker.smoothing + 0.05)
                ui.toast(f"Smoothing: {tracker.smoothing:.2f}")

            # Mode-specific controls
            if mode == "classic":
                if key == ord("w"):
                    detector.saturation_max = max(0, detector.saturation_max - 5)
                    print(f"[saturation_max = {detector.saturation_max}]")
                elif key == ord("e"):
                    detector.saturation_max = min(255, detector.saturation_max + 5)
                    print(f"[saturation_max = {detector.saturation_max}]")
                elif key in (ord("+"), ord("=")):
                    detector.bright_threshold = min(255, detector.bright_threshold + 5)
                    print(f"[bright_threshold = {detector.bright_threshold}]")
                elif key in (ord("-"), ord("_")):
                    detector.bright_threshold = max(50, detector.bright_threshold - 5)
                    print(f"[bright_threshold = {detector.bright_threshold}]")
                elif key == ord("["):
                    detector.motion_threshold = max(2, detector.motion_threshold - 2)
                    print(f"[motion_threshold = {detector.motion_threshold}]")
                elif key == ord("]"):
                    detector.motion_threshold = min(80, detector.motion_threshold + 2)
                    print(f"[motion_threshold = {detector.motion_threshold}]")
            else:
                # Deep Learning modes (TrackNet / YOLO): [/] adjusts confidence
                if mode == "tracknet" and hasattr(detector, "tracknet") and detector.tracknet:
                    if key == ord("["):
                        detector.tracknet.confidence = max(0.05, detector.tracknet.confidence - 0.05)
                        print(f"[tracknet confidence = {detector.tracknet.confidence:.2f}]")
                    elif key == ord("]"):
                        detector.tracknet.confidence = min(0.95, detector.tracknet.confidence + 0.05)
                        print(f"[tracknet confidence = {detector.tracknet.confidence:.2f}]")
                elif mode == "hybrid" and hasattr(detector, "yolo") and detector.yolo:
                    if key == ord("["):
                        new_c = max(0.05, detector.yolo._confidence - 0.05)
                        detector.yolo._confidence = new_c
                        print(f"[yolo confidence = {new_c:.2f}]")
                    elif key == ord("]"):
                        new_c = min(0.95, detector.yolo._confidence + 0.05)
                        detector.yolo._confidence = new_c
                        print(f"[yolo confidence = {new_c:.2f}]")

    except KeyboardInterrupt:
        pass
    finally:
        # Save configuration before exit
        config.set("smoothing", tracker.smoothing)
        config.set("show_minimap", visualizer.show_minimap)
        config.set("style", visualizer.style)
        config.set("trail_opacity", visualizer.trail_opacity)
        config.save()

        # A session still recording at exit would otherwise be thrown away,
        # including on Ctrl-C.
        if recorder.is_recording:
            saved = recorder.stop_recording(tracker)
            if saved:
                print(f"[tracker] session written to {saved}")

        cv2.destroyAllWindows()
        rallies = tracker.rallies_completed
        summary = (f"\nTracker stopped. "
                   f"Total hits this session: {tracker.hit_count}, "
                   f"rallies: {rallies}")
        if tracker.rally_lengths:
            longest = max(tracker.rally_lengths)
            mean = sum(tracker.rally_lengths) / len(tracker.rally_lengths)
            summary += f", longest {longest} hits, mean {mean:.1f}"
        print(summary + ".")


def _build_hud(detector, tracker, mode, det_source, fps):
    """Build the HUD extra string based on detector mode."""
    if mode == "classic":
        det_speed = float(np.hypot(*detector.get_velocity()))
        lock = "LOCK" if detector.locked else "----"
        return (f"bright:{detector.bright_threshold} "
                f"sat:{detector.saturation_max} "
                f"motion:{detector.motion_threshold} "
                f"cand:{detector.last_num_candidates} "
                f"[{detector.last_stage}] "
                f"{lock} conf:{detector.get_confidence():.1f} "
                f"v:{det_speed:4.1f} "
                f"smooth:{tracker.smoothing:.2f}")
    elif mode == "tracknet":
        src = det_source or "none"
        tn = getattr(detector, "tracknet", None)
        tn_conf = tn.confidence if tn is not None else 0
        tn_on = "ON" if tn is not None else "OFF"
        res = (f"{tn.input_width}x{tn.input_height}" if tn is not None else "-")
        fp16 = "16" if tn is not None and tn.fp16 else "32"
        engine = getattr(detector, 'engine_name', 'unknown')
        return (f"[TrackNet {tn_on} {engine}] src:{src} "
                f"conf:{tn_conf:.2f} res:{res} fp{fp16} "
                f"smooth:{tracker.smoothing:.2f} "
                f"fps:{fps:.0f}")
    else:
        src = det_source or "none"
        yolo_conf = getattr(detector.yolo, "_confidence", 0)
        yolo_on = "ON" if getattr(detector, "yolo", None) else "OFF"
        return (f"[YOLO {yolo_on}] src:{src} "
                f"conf:{yolo_conf:.2f} "
                f"smooth:{tracker.smoothing:.2f} "
                f"fps:{fps:.0f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
