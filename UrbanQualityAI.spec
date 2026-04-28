# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


hiddenimports = []

# Qt WebEngine and friends are notorious for dynamic imports; help PyInstaller a bit.
hiddenimports += collect_submodules("PySide6.QtWebEngineCore")
hiddenimports += collect_submodules("PySide6.QtWebEngineWidgets")
# pyogrio uses compiled extension modules loaded dynamically.
hiddenimports += collect_submodules("pyogrio")


a = Analysis(
    ["src\\uq_desktop_processor\\__main__.py"],
    pathex=["src"],
    binaries=[
        # Bundle GDAL-related native libs shipped with the pyogrio wheel.
        *collect_dynamic_libs("pyogrio"),
    ],
    datas=[
        ("src\\uq_desktop_processor\\assets", "uq_desktop_processor\\assets"),
        # openai/CLIP tokenizer vocab (and other clip assets) must be bundled as data files,
        # otherwise the packaged app will crash trying to open `bpe_simple_vocab_16e6.txt.gz`.
        *collect_data_files("clip"),
        # Include pyogrio package data files required by GDAL/OGR runtime.
        *collect_data_files("pyogrio"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # hook-torch tries to import tensorboard via torch.utils.tensorboard;
        # exclude it unless you explicitly depend on TensorBoard.
        "tensorboard",
        "torch.utils.tensorboard",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="uq_desktop_processor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["src\\uq_desktop_processor\\assets\\img\\icon.ico"],
)

