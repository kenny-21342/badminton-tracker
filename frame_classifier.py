"""
Frame classifier for badminton broadcast video.

Identifies which frames are suitable for shuttlecock detection/training
by analyzing visual features. Filters out:
  - Player close-ups (face/torso fills frame, no court visible)
  - Crowd / audience shots
  - Slow-motion replays (low inter-frame motion, different color grading)
  - Hawk-Eye / 3D graphics (synthetic green/blue court overlays)
  - Scorebug-only frames (graphics, no gameplay)

Usage:
    from frame_classifier import classify_frame, FrameType

    ftype, confidence = classify_frame(frame)
    if ftype == FrameType.LIVE_COURT:
        # good frame for detection/training
"""
import cv2
import numpy as np
from enum import Enum


class FrameType(Enum):
    LIVE_COURT = "live_court"        # Normal rally footage — keep
    PLAYER_CLOSEUP = "closeup"       # Player face/torso — skip
    CROWD = "crowd"                  # Audience shot — skip
    REPLAY = "replay"                # Slow-motion replay — skip
    HAWKEYE = "hawkeye"             # 3D graphics overlay — skip
    GRAPHICS = "graphics"           # Scorebug/title card only — skip
    UNKNOWN = "unknown"


def classify_frame(frame, prev_gray=None):
    """
    Classify a broadcast frame. Returns (FrameType, confidence 0-1).

    Args:
        frame:     BGR image (any size).
        prev_gray: Optional previous-frame grayscale (downscaled) for
                   motion-based replay detection. Pass None to skip.
    """
    h, w = frame.shape[:2]
    scale = 480.0 / max(h, w)
    small = cv2.resize(frame, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    scores = {}

    # --- Hawk-Eye / 3D graphics detection -------------------------------
    scores[FrameType.HAWKEYE] = _detect_hawkeye(hsv)

    # --- Crowd detection ------------------------------------------------
    scores[FrameType.CROWD] = _detect_crowd(small, hsv)

    # --- Player close-up detection --------------------------------------
    scores[FrameType.PLAYER_CLOSEUP] = _detect_closeup(small, hsv)

    # --- Graphics / title card detection --------------------------------
    scores[FrameType.GRAPHICS] = _detect_graphics(small, gray)

    # --- Replay detection (needs previous frame) ------------------------
    if prev_gray is not None:
        scores[FrameType.REPLAY] = _detect_replay(gray, prev_gray, hsv)
    else:
        scores[FrameType.REPLAY] = 0.0

    # --- Live court -----------------------------------------------------
    # Court view is the strongest signal — if we see multiple long lines
    # in multiple orientations, it's almost certainly a court view.
    court_score = _detect_court_view(gray)
    if court_score > 0.4:
        # Strong court structure present. Override lower-scoring categories
        # unless they scored very high (>0.6).
        for ft in scores:
            if scores[ft] < 0.6:
                scores[ft] *= 0.3
        scores[FrameType.LIVE_COURT] = court_score
    else:
        scores[FrameType.LIVE_COURT] = court_score * 0.2

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # If nothing scored confidently, return UNKNOWN rather than a weak guess.
    if best_score < 0.4 and best_type != FrameType.LIVE_COURT:
        return FrameType.UNKNOWN, best_score

    return best_type, best_score


# ----------------------------------------------------------------- classifiers
def _detect_court_view(gray):
    """
    Court-line detection. Returns 0-1 confidence that this frame
    shows a real badminton court.

    A real court view has:
      - Multiple long horizontal lines (service lines, net line,
        baseline) SPREAD across the vertical axis of the frame.
      - Optionally vertical/diagonal lines (sidelines, center line)
        but these are often broken by perspective + compression, so
        we don't strictly require them.
    """
    _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                              np.ones((2, 2), np.uint8))

    lines = cv2.HoughLinesP(
        bright, rho=1, theta=np.pi / 180, threshold=35,
        minLineLength=int(gray.shape[1] * 0.12),
        maxLineGap=15,
    )
    if lines is None or len(lines) < 3:
        return 0.0

    # Classify line orientations. Use generous bins because broadcast
    # perspective makes court lines appear slightly angled.
    horiz_lines = []
    other_lines = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 35 or angle > 145:
            horiz_lines.append((y1, y2, x1, x2))
        else:
            other_lines.append((x1, x2, y1, y2, angle))

    if len(horiz_lines) < 2:
        return 0.0

    # --- Spatial spread check (vertical) --------------------------------
    # Court service lines are at different vertical positions. A close-up
    # or scorebug has lines concentrated at one height.
    h_y_centers = [(y1 + y2) / 2 for y1, y2, _, _ in horiz_lines]
    y_spread = max(h_y_centers) - min(h_y_centers) if len(h_y_centers) > 1 else 0
    frame_h, frame_w = gray.shape

    if y_spread < frame_h * 0.2:
        return 0.0

    # --- Scoring ---------------------------------------------------------
    # More horizontal lines + good spread + some non-horizontal lines
    # = higher confidence.
    score = min(0.6, len(horiz_lines) / 15.0)
    if y_spread > frame_h * 0.35:
        score += 0.2
    if len(other_lines) >= 2:
        score += 0.2

    return min(1.0, score)


def _detect_hawkeye(hsv):
    """
    Hawk-Eye frames are 3D-rendered courts with saturated artificial
    colors: bright green or blue surfaces, high saturation, very
    uniform texture. Real courts have lower, more varied saturation.
    """
    h, s, v = cv2.split(hsv)
    mean_s = float(np.mean(s))

    green_mask = cv2.inRange(h, 35, 85)
    blue_mask = cv2.inRange(h, 85, 130)
    saturated_green = np.sum((green_mask > 0) & (s > 60))
    saturated_blue = np.sum((blue_mask > 0) & (s > 60))
    total_px = h.shape[0] * h.shape[1]
    vivid_ratio = (saturated_green + saturated_blue) / total_px

    # Hawk-Eye renders are smooth (low texture). Camera footage is noisy.
    # This is the KEY discriminator — real camera footage has lap_var > 500,
    # while 3D renders are typically < 200.
    gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap))

    # Must be smooth to be Hawk-Eye. Noisy camera footage is never Hawk-Eye.
    if lap_var > 400:
        return 0.0

    score = 0.0
    if vivid_ratio > 0.3:
        score += 0.4
    if lap_var < 200:
        score += 0.4
    if vivid_ratio > 0.45 and lap_var < 200:
        score += 0.2

    return min(1.0, score)


def _detect_crowd(frame, hsv):
    """
    Crowd shots have very high texture/detail variance (hundreds of
    small heads) and NO long straight court lines. Also tends to be
    darker and more uniform than a lit court.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap))

    # Crowd shots are dark (no brightly lit court surface).
    mean_v = float(np.mean(cv2.split(hsv)[2]))

    # If there are long straight lines, it's a court, not a crowd.
    _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(bright, 1, np.pi / 180, 35,
                            minLineLength=int(gray.shape[1] * 0.12),
                            maxLineGap=15)
    n_lines = 0 if lines is None else len(lines)
    if n_lines > 5:
        return 0.0  # has court structure

    score = 0.0
    if lap_var > 800 and mean_v < 130:
        score += 0.3
    if mean_v < 100:
        score += 0.2
    # High edge density but random orientation (not structured lines).
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.sum(edges > 0)) / (gray.shape[0] * gray.shape[1])
    if edge_ratio > 0.15 and n_lines < 3:
        score += 0.2

    return min(1.0, score)


def _detect_closeup(frame, hsv):
    """
    Player close-ups: large skin-tone regions, no court lines, shallow
    depth of field (background blurred).
    """
    h, s, v = cv2.split(hsv)
    # Skin tone in HSV: H 0-25, S 30-150, V >= 60.
    skin = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 170, 255]))
    skin_ratio = float(np.sum(skin > 0)) / (h.shape[0] * h.shape[1])

    score = 0.0
    if skin_ratio > 0.08:
        score += 0.5
    if skin_ratio > 0.15:
        score += 0.3

    # No long straight lines (court lines absent in close-ups).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(bright, 1, np.pi / 180, 30,
                            minLineLength=40, maxLineGap=10)
    n_lines = 0 if lines is None else len(lines)
    if n_lines <= 1 and skin_ratio > 0.05:
        score += 0.2

    return min(1.0, score)


def _detect_graphics(frame, gray):
    """
    Pure graphics frames (title cards, full-screen scorebugs, ads).
    Very uniform large-color-block regions AND no court lines AND no skin.
    """
    # Must NOT have court lines (otherwise it's a court view that happens
    # to be smooth).
    _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(bright, 1, np.pi / 180, 30,
                            minLineLength=40, maxLineGap=10)
    n_lines = 0 if lines is None else len(lines)
    if n_lines > 2:
        return 0.0  # has court structure — not pure graphics

    # Low texture.
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap))

    # Flat color blocks.
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    diff = np.max(np.abs(frame.astype(np.int16) - blur.astype(np.int16)),
                  axis=2)
    flat_ratio = float(np.sum(diff < 8)) / diff.size

    score = 0.0
    if lap_var < 200 and n_lines <= 1:
        score += 0.3
    if flat_ratio > 0.8 and n_lines <= 1:
        score += 0.5
    if flat_ratio > 0.9:
        score += 0.2

    return min(1.0, score)


def _detect_replay(gray, prev_gray, hsv):
    """
    Slow-motion replays differ from live footage in two ways:
      1. Lower inter-frame motion (slow-mo = less movement per frame).
      2. Different color grading (often slightly washed-out / tinted).

    This only works when prev_gray is provided (consecutive frames).
    Without it, returns 0.
    """
    # Inter-frame motion magnitude.
    diff = cv2.absdiff(gray, prev_gray)
    motion = float(np.mean(diff))

    # Replay frames have very low motion but aren't static graphics either.
    score = 0.0
    if 1.0 < motion < 4.0:
        # Low motion = possible slow-mo (but could also be a paused rally).
        score += 0.3

    # Color grading: replays often have a slight tint or lower saturation.
    h, s, v = cv2.split(hsv)
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))
    if mean_v > 180 and mean_s < 40:
        # Washed-out / bright = replay signature.
        score += 0.3
        if 1.0 < motion < 4.0:
            score += 0.4

    return min(1.0, score)
