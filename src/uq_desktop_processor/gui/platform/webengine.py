"""
Qt WebEngine profile and view helpers for embedding maps and web content.
"""

import logging
import os
import sys


def configure_webengine_for_deck_gl() -> None:
    """
    deck.gl needs a working **WebGL** context. ``--disable-gpu`` turns WebGL off entirely
    (GL_RENDERER = Disabled), so the map cannot start.

    On Windows, native ANGLE/D3D11 can hit DXGI_ERROR_DEVICE_REMOVED; the stable default is
    **SwiftShader** (CPU WebGL via ``--use-angle=swiftshader``).

    If ``QTWEBENGINE_CHROMIUM_FLAGS`` is already set, it is left unchanged.

    ``CITYLENS_WEBENGINE_USE_GPU=1``: try native GPU with light sandbox mitigations only.
    """
    if os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip():
        return
    if sys.platform != "win32":
        return
    log = logging.getLogger(__name__)
    if os.environ.get("CITYLENS_WEBENGINE_USE_GPU", "").lower() in ("1", "true", "yes"):
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu-sandbox --ignore-gpu-blocklist"
        log.info(
            "Qt WebEngine: native GPU (no SwiftShader). If the map crashes, unset "
            "CITYLENS_WEBENGINE_USE_GPU or set flags manually via QTWEBENGINE_CHROMIUM_FLAGS."
        )
        return
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--use-angle=swiftshader " "--enable-unsafe-swiftshader " "--disable-gpu-sandbox " "--ignore-gpu-blocklist"
    )
    log.info("Qt WebEngine: SwiftShader (CPU WebGL) for deck.gl. For native GPU: " "set CITYLENS_WEBENGINE_USE_GPU=1")
