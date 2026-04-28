"""
Shared GUI filesystem paths.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
APP_ICON_PATH = PACKAGE_ROOT / "assets" / "img" / "icon.ico"

__all__ = ["PROJECT_ROOT", "PACKAGE_ROOT", "APP_ICON_PATH"]
