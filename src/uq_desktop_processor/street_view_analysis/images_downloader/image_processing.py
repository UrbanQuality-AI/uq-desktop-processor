"""
Resize, EXIF, and panorama-to-perspective conversion for downloaded images.
"""

import logging
from datetime import UTC, datetime

import cv2
import numpy as np
import piexif
import py360convert
import requests
from PIL import Image

log = logging.getLogger(__name__)


def _decimal_deg_to_exif_rationals(decimal_deg: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Convert signed decimal degrees to EXIF GPS DMS rationals (absolute value; Ref gives hemisphere).

    :param decimal_deg: Latitude or longitude in degrees.
    :return: Three rationals ``(degrees, minutes, seconds)`` for piexif.

    Example::
        In: _decimal_deg_to_exif_rationals(50.5)
        Out: ((50, 1), (30, 1), (0, 1000000))
    """
    absolute = abs(decimal_deg)
    deg = int(absolute)
    minutes_float = (absolute - deg) * 60.0
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60.0
    sec_num = int(round(seconds * 1_000_000))
    # Keep sub-second rational within one minute
    if sec_num >= 60 * 1_000_000:
        sec_num = 59_999_999
    return (deg, 1), (minutes, 1), (sec_num, 1_000_000)


def save_jpeg_with_exif(
    bgr: np.ndarray,
    path: str,
    *,
    lat: float,
    lon: float,
    captured_at_ms: int | float | None = None,
    jpeg_quality: int = 95,
) -> None:
    """
    Save a BGR image as JPEG with GPS and optional capture time in EXIF.

    :param bgr: Image in OpenCV BGR layout.
    :param path: Output ``.jpg`` path.
    :param lat: Latitude in WGS84 (degrees).
    :param lon: Longitude in WGS84 (degrees).
    :param captured_at_ms: Unix time in milliseconds (e.g. Mapillary ``captured_at``), or None.
    :param jpeg_quality: JPEG quality 1-100.

    Example::
        In: save_jpeg_with_exif(img, "out.jpg", lat=50.06, lon=19.94, captured_at_ms=1710000000000)
        Out: "out.jpg" saved with GPS EXIF and capture timestamp metadata
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # piexif expects these IFD buckets even when mostly empty
    exif_dict: dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    if captured_at_ms is not None:
        try:
            ts = float(captured_at_ms) / 1000.0
            dt = datetime.fromtimestamp(ts, tz=UTC)
            dt_str = dt.strftime("%Y:%m:%d %H:%M:%S")
        except (OSError, OverflowError, TypeError, ValueError):
            dt_str = None
        if dt_str:
            exif_dict["0th"][piexif.ImageIFD.DateTime] = dt_str
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt_str
            exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = dt_str

    # GPS refs encode hemisphere; rationals use absolute values
    lat_ref = b"N" if lat >= 0 else b"S"
    lon_ref = b"E" if lon >= 0 else b"W"
    exif_dict["GPS"][piexif.GPSIFD.GPSVersionID] = (2, 0, 0, 0)
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = lat_ref
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = _decimal_deg_to_exif_rationals(lat)
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = lon_ref
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = _decimal_deg_to_exif_rationals(lon)

    exif_bytes = piexif.dump(exif_dict)
    pil_image.save(path, format="JPEG", quality=jpeg_quality, exif=exif_bytes)


def download_image(url: str, timeout: float) -> np.ndarray | None:
    """
    Download an image URL and decode it to a BGR ``ndarray``.

    :param url: Image URL.
    :param timeout: HTTP timeout in seconds.
    :return: BGR image, or None on network/decode failure.

    Example::
        In: download_image("https://example.com/photo.jpg", timeout=10.0)
        Out: decoded BGR image array (H, W, 3) or None if download/decode fails
    """
    try:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Downloading image from: %s", url)

        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        # OpenCV expects a 1-D uint8 buffer for imdecode
        image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            log.warning("OpenCV failed to decode image from: %s", url)
            return None

        return image

    except (requests.RequestException, ValueError) as download_error:
        log.warning("Download/decode error for %s: %s", url, download_error)
        return None


def process_panorama(img: np.ndarray, fov_deg: float, out_hw: tuple[int, int]) -> np.ndarray:
    """
    Convert a 360° equirectangular image to a perspective crop.

    Scalar ``fov_deg`` is expanded to ``(h_fov, v_fov)`` so py360convert uses a stable code path.

    :param img: Equirectangular panorama (BGR).
    :param fov_deg: Field of view in degrees, or ``(h, v)`` if a pair is needed.
    :param out_hw: Output ``(height, width)``.
    :return: Perspective image as BGR ``ndarray``.

    Example::
        In: process_panorama(pano_bgr, fov_deg=90, out_hw=(512, 512))
        Out: perspective BGR image array with shape (512, 512, 3)
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "Converting panorama (Input shape: %s, FOV: %s)",
            img.shape,
            fov_deg,
        )

    # Tuple FOV avoids a buggy scalar branch in some py360convert versions
    fov_hv = (
        (float(fov_deg[0]), float(fov_deg[1]))
        if isinstance(fov_deg, tuple | list) and len(fov_deg) >= 2
        else (float(fov_deg), float(fov_deg))
    )

    return py360convert.e2p(
        img,
        fov_hv,
        0,
        0,
        out_hw,
    )
