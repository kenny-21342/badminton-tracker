"""
Screen Recording permission preflight for macOS.

macOS does not report a screen-capture denial as an error. `mss.grab` succeeds
and hands back a correctly sized frame that is simply blank -- every pixel the
same colour. Without a check the tracker runs happily against that blank frame,
detects nothing, and looks like a broken detector rather than a missing
permission.

The one signal we get is the frame content, so that is what we test.
"""
import sys

import numpy as np

# A real desktop, even a solid-colour wallpaper with a menu bar, clears this by
# orders of magnitude. A denied capture gives exactly 1.
MIN_DISTINCT_COLOURS = 4


def _pack(frame):
    """Fold each pixel's channels into one integer, so colours compare as scalars."""
    array = np.ascontiguousarray(frame)
    flat = array.reshape(-1, array.shape[-1])
    if flat.dtype == np.uint8 and flat.shape[1] == 4:
        return flat.view(np.uint32).ravel()
    packed = np.zeros(len(flat), dtype=np.uint64)
    for channel in range(flat.shape[1]):
        packed = (packed << np.uint64(8)) | flat[:, channel].astype(np.uint64)
    return packed


def count_distinct_colours(frame, limit=MIN_DISTINCT_COLOURS):
    """Distinct pixel values in `frame`, counted no further than `limit`.

    Scans every pixel but stops as soon as `limit` distinct values have been
    seen, which is all the caller needs to know. That bound is what makes an
    exact count affordable: np.unique over a 1920x1080 grab costs ~900 ms,
    while this settles in a few milliseconds because a real desktop hits the
    limit almost immediately.

    Counting every pixel rather than a subsample matters. A sampled count
    misses small patches of content -- a menu bar clock on an otherwise empty
    desktop -- and would call a permitted capture a denial.
    """
    packed = _pack(frame)
    seen = 0
    while packed.size and seen < limit:
        packed = packed[packed != packed[0]]
        seen += 1
    return seen


def frame_is_blank(frame):
    """Whether `frame` looks like a denied capture rather than a desktop."""
    return count_distinct_colours(frame) < MIN_DISTINCT_COLOURS


def host_app_description():
    """How to describe, to the user, the app that needs the permission."""
    if getattr(sys, "frozen", False):
        return "Badminton Tracker"
    return "the terminal app you launched this from"


def permission_message():
    """The text shown when a capture comes back blank."""
    return (
        "\n"
        "Screen Recording permission looks denied.\n"
        "\n"
        "macOS returned a blank frame instead of your screen, which is what it\n"
        "does when the capture is not permitted. The tracker would run, but it\n"
        "would see nothing at all.\n"
        "\n"
        f"To fix: System Settings -> Privacy & Security -> Screen Recording,\n"
        f"enable {host_app_description()}, then quit and reopen it.\n"
        "The permission does not take effect until the app is restarted.\n"
        "\n"
        "If you have already done this, restart the app once more -- macOS\n"
        "applies the grant only to freshly launched processes.\n"
    )


def report_problem(message):
    """Put `message` somewhere the user will actually see it.

    A double-clicked .app has no terminal attached, so printing is not enough:
    the app would appear to do nothing at all. When frozen, also raise a native
    alert. Printing stays unconditional so the source checkout is unchanged.
    """
    print(message, file=sys.stderr)
    if not getattr(sys, "frozen", False):
        return
    import subprocess

    # osascript rather than a Tk or Qt dialog: no extra dependency, and it
    # cannot be affected by whatever state the capture failure left behind.
    script = (
        'display alert "Screen Recording permission needed" '
        f'message {_applescript_string(message.strip())} as critical'
    )
    try:
        subprocess.run(["osascript", "-e", script], timeout=120, check=False)
    except Exception:
        pass  # The alert is a courtesy; never let it mask the real problem.


def _applescript_string(text):
    """Quote `text` as an AppleScript string literal."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped.replace("\n", '" & return & "') + '"'


def check_capture_permitted(sct, monitor):
    """Grab one frame and report whether the capture is really permitted.

    Returns (ok, message). `message` is None when ok.
    """
    try:
        frame = np.array(sct.grab(monitor))
    except Exception as exc:  # pragma: no cover - platform dependent
        return False, (f"\nScreen capture failed: {exc}\n"
                       f"{permission_message()}")
    if frame.size == 0 or frame_is_blank(frame):
        return False, permission_message()
    return True, None
