"""
Interactive region selector. Shows a frozen screenshot of the chosen monitor
and lets you drag a rectangle around the badminton court. Returns the region
as a dict suitable for mss: {"top": y, "left": x, "width": w, "height": h}.

Usage:
    from region_selector import select_region
    region = select_region(monitor_index=1)
"""
import cv2
import numpy as np
from mss import mss


class _DragState:
    def __init__(self):
        self.start = None
        self.end = None
        self.dragging = False

    def reset(self):
        self.start = None
        self.end = None
        self.dragging = False


def select_region(monitor_index=1):
    """
    Open a window with a snapshot of the monitor. Drag to select a region.
    Press ENTER / SPACE to confirm, ESC to cancel (returns full monitor).
    """
    sct = mss()
    if monitor_index < 0 or monitor_index >= len(sct.monitors):
        monitor_index = 1
    mon = sct.monitors[monitor_index]

    shot = sct.grab(mon)
    img = np.array(shot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Scale down for display if the monitor is huge (e.g. 4K)
    disp_h = 900
    scale = 1.0
    if img.shape[0] > disp_h:
        scale = disp_h / img.shape[0]
        disp = cv2.resize(img, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    else:
        disp = img.copy()

    state = _DragState()
    win = "Drag to select court region - ENTER to confirm, ESC to use full screen"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.start = (x, y)
            state.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state.end = (x, y)
            state.dragging = False

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("\n[region selector] Drag a box around the court area, then press ENTER.")
    print("[region selector] Press ESC to capture the whole monitor instead.\n")

    while True:
        canvas = disp.copy()
        if state.start and state.end:
            cv2.rectangle(canvas, state.start, state.end, (0, 255, 255), 2)
            cv2.rectangle(canvas, state.start, state.end, (0, 0, 0), 1)
        # Instructions
        cv2.putText(canvas, "Drag a box around the court. ENTER=confirm  ESC=full",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(15) & 0xFF
        if key in (13, 32):  # Enter or Space
            break
        if key == 27:  # Esc
            state.reset()
            break

    cv2.destroyAllWindows()

    if state.start and state.end:
        x1 = int(min(state.start[0], state.end[0]) / scale)
        y1 = int(min(state.start[1], state.end[1]) / scale)
        x2 = int(max(state.start[0], state.end[0]) / scale)
        y2 = int(max(state.start[1], state.end[1]) / scale)
        # Translate back into absolute screen coordinates
        region = {
            "left": mon["left"] + x1,
            "top": mon["top"] + y1,
            "width": x2 - x1,
            "height": y2 - y1,
        }
    else:
        region = {"left": mon["left"], "top": mon["top"],
                  "width": mon["width"], "height": mon["height"]}

    print(f"[region selector] Selected region: {region['width']}x{region['height']} "
          f"at ({region['left']}, {region['top']})")
    return region


if __name__ == "__main__":
    # Standalone: just run the selector on the primary monitor.
    r = select_region(1)
    print(r)
