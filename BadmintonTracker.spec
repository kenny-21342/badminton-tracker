# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the double-clickable macOS app.

Build with:   python3 -m PyInstaller --noconfirm BadmintonTracker.spec
Result:       dist/Badminton Tracker.app

The TrackNet weights are bundled (130 MB) so the app works on first launch
without a download. Torch comes along because tracknet_detector imports it at
module scope; that is most of the bundle size.
"""

import os

block_cipher = None


def mlx_lib_dir():
    """Path to mlx/lib, or None when mlx is not installed."""
    try:
        import mlx
    except ImportError:
        return None
    # mlx is a namespace package, so __file__ is None; __path__ is the truth.
    for root in list(getattr(mlx, '__path__', [])):
        path = os.path.join(root, 'lib')
        if os.path.isdir(path):
            return path
    return None


def mlx_support_files():
    """Everything in mlx/lib, as (binaries, datas).

    PyInstaller collects libmlx.dylib because core.*.so links it directly, but
    stops there: it misses libjaccl.dylib, which libmlx loads through @rpath,
    and mlx.metallib, which is data rather than a library. Missing either one
    makes `import mlx.core` fail inside the bundle, and the engine picker then
    falls back to the slower Torch backend without saying why.

    Returns ([], []) when mlx is absent so the build still works without it.
    """
    lib = mlx_lib_dir()
    if lib is None:
        return [], []
    binaries, datas = [], []
    for name in sorted(os.listdir(lib)):
        path = os.path.join(lib, name)
        if not os.path.isfile(path):
            continue
        (binaries if name.endswith('.dylib') else datas).append(
            (path, 'mlx/lib'))
    return binaries, datas


MLX_BINARIES, MLX_DATAS = mlx_support_files()

a = Analysis(
    ['main_screen.py'],
    pathex=[],
    binaries=MLX_BINARIES,
    datas=[
        # Looked up through resources.candidate_paths at runtime.
        ('models/TrackNet_best.pt', 'models'),
    ] + MLX_DATAS,
    hiddenimports=[
        # Imported lazily inside create_detector, so the analyser misses them.
        'tracknet_detector',
        'mlx_tracknet_detector',
        # mlx is a namespace package whose submodules the analyser cannot
        # follow; without these the MLX engine fails to import in the bundle
        # and TrackNet silently drops to the slower Torch backend.
        'mlx',
        'mlx.core',
        'mlx.nn',
        'mlx.nn.layers',
        'mlx.utils',
        'detector',
        'yolo_detector',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Never imported by the capture app, and each drags in a large tree.
        'matplotlib',
        'pandas',
        'polars',
        'scipy',
        'tkinter',
        'PIL.ImageQt',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'coremltools',
        'ultralytics',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BadmintonTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='BadmintonTracker',
)

app = BUNDLE(
    coll,
    name='Badminton Tracker.app',
    icon=None,
    bundle_identifier='bond.busypizza.badmintontracker',
    info_plist={
        'CFBundleName': 'Badminton Tracker',
        'CFBundleDisplayName': 'Badminton Tracker',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        # Without this the app has no Dock icon and cannot take keyboard focus,
        # so the OpenCV window would never receive the control keys.
        'LSBackgroundOnly': False,
        'NSHighResolutionCapable': True,
        # macOS shows this in the Screen Recording permission prompt.
        'NSCameraUsageDescription':
            'Badminton Tracker reads video frames to follow the shuttlecock.',
    },
)
