"""Finding bundled data files whether running from source or from a .app.

Model weights are looked up by relative path ("models/TrackNet_best.pt"), which
works from a source checkout because the working directory is the repo. A
double-clicked .app is launched with the working directory set to "/", so the
same relative path finds nothing and TrackNet silently falls back to the
classic detector -- the one measured at 0.4% recall.
"""
import os
import sys


def resource_root():
    """The directory bundled data files live in."""
    # PyInstaller unpacks data files here and sets sys.frozen.
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return bundled
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(relative):
    """Absolute path to a bundled data file."""
    return os.path.join(resource_root(), relative)


def candidate_paths(relative_paths):
    """Each path as given and as resolved against the bundle.

    The as-given form is kept first so a file in the working directory still
    wins -- that is how someone points the app at their own weights.
    """
    resolved = []
    for path in relative_paths:
        resolved.append(path)
        absolute = resource_path(path)
        if absolute not in resolved:
            resolved.append(absolute)
    return resolved
