"""
Parses lat/lon from image metadata and misc helpers for point layer creation.
"""

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _exif_gps_to_decimal(exif_gps: dict[int, Any]) -> tuple[float, float] | None:
    """
    Convert EXIF GPSInfo dict to (lat, lon) in decimal degrees.

    Works with EXIF dicts returned by PIL.Image.getexif() where GPSInfo is a mapping
    of numeric keys to values (rationals, tuples).

    :param exif_gps: GPSInfo mapping from EXIF.
    :return: ``(lat, lon)`` if GPS fields are complete and valid, otherwise None.

    Example::
        In: _exif_gps_to_decimal({1: "N", 2: ((50,1),(3,1),(0,1)), 3: "E", 4: ((19,1),(56,1),(0,1))})
        Out: (50.05, 19.933333333333334)
    """

    def _ratio_to_float(ratio_value: Any) -> float:
        # PIL can return IFDRational or tuple(num, den)
        try:
            return float(ratio_value)
        except (TypeError, ValueError):
            pass
        if isinstance(ratio_value, tuple | list) and len(ratio_value) == 2:
            numerator, denominator = ratio_value
            return float(numerator) / float(denominator)
        raise ValueError(f"Unsupported rational type: {type(ratio_value)}")

    def _dms_to_deg(dms_value: Any, reference: str) -> float:
        if not isinstance(dms_value, tuple | list) or len(dms_value) != 3:
            raise ValueError("GPS DMS must be a 3-tuple")
        degrees = _ratio_to_float(dms_value[0])
        minutes = _ratio_to_float(dms_value[1])
        seconds = _ratio_to_float(dms_value[2])
        decimal_degrees = degrees + minutes / 60.0 + seconds / 3600.0
        normalized_reference = (reference or "").upper()
        if normalized_reference in ("S", "W"):
            decimal_degrees = -decimal_degrees
        return float(decimal_degrees)

    try:
        # EXIF GPS tag ids: 1/2 latitude ref+value, 3/4 longitude ref+value.
        lat_ref = exif_gps.get(1)  # GPSLatitudeRef
        lat_dms = exif_gps.get(2)  # GPSLatitude
        lon_ref = exif_gps.get(3)  # GPSLongitudeRef
        lon_dms = exif_gps.get(4)  # GPSLongitude
        if not lat_ref or not lon_ref or not lat_dms or not lon_dms:
            return None
        lat = _dms_to_deg(lat_dms, str(lat_ref))
        lon = _dms_to_deg(lon_dms, str(lon_ref))
        return lat, lon
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def parse_lat_lon_from_image_metadata(image_path: str | Path) -> tuple[float, float]:
    """
    Extract (lat, lon) from image EXIF GPS metadata.

    :param image_path: Path to an image file.
    :return: ``(latitude, longitude)`` from EXIF GPS tags.
    :raises ValueError: if GPS metadata is missing or invalid.

    Example::
        In: parse_lat_lon_from_image_metadata("data/images/raw/123.jpg")
        Out: (50.0612, 19.9377)
    """
    from PIL import Image

    image_path_obj = Path(image_path)
    try:
        with Image.open(image_path_obj) as im:
            exif = im.getexif()
            if not exif:
                raise ValueError("missing EXIF")
            gps_info = exif.get_ifd(0x8825)  # GPSInfo IFD
            if not isinstance(gps_info, dict):
                raise ValueError("missing GPSInfo")
            latlon_coordinates = _exif_gps_to_decimal(gps_info)
            if latlon_coordinates is None:
                raise ValueError("missing GPS lat/lon")
            return latlon_coordinates
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"cannot read EXIF GPS from image: {image_path_obj}") from error


def parse_lat_lon(image_path: str | Path) -> tuple[float, float]:
    """
    Parse coordinates strictly from EXIF GPS metadata.

    :param image_path: Path to an image file.
    :return: ``(latitude, longitude)``.

    Example::
        In: parse_lat_lon("data/images/raw/img_0001.jpg")
        Out: (50.0612, 19.9377)
    """
    return parse_lat_lon_from_image_metadata(image_path)


def _is_geopandas_available() -> bool:
    """
    Check whether GeoPandas and Shapely are available in the environment.

    :return: True if both packages can be imported, False otherwise.

    Example::
        In: _is_geopandas_available()
        Out: True
    """
    try:
        import geopandas  # noqa: F401
        import shapely  # noqa: F401

        return True
    except ImportError as error:
        log.debug("GeoPandas availability check failed: %s", error)
        return False
