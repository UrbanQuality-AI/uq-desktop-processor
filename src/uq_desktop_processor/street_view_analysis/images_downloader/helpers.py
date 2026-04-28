"""
Shared helpers for paths, HTTP sessions, and image naming in the downloader.
"""

import re


def mapillary_image_filename(image_id: str) -> str:
    """
    Build a filesystem-safe JPEG filename from a Mapillary image id.

    :param image_id: Mapillary image identifier (as returned by the Graph API).
    :return: Filename such as ``{id}.jpg``.

    Example::
        In: mapillary_image_filename("123/abc")
        Out: "123_abc.jpg"
    """
    # Strip unsafe path characters for cross-platform filenames
    sanitized_image_id = re.sub(r"[^\w.\-]", "_", str(image_id).strip())
    if not sanitized_image_id:
        raise ValueError("image_id must be non-empty after sanitization")
    return f"{sanitized_image_id}.jpg"


def lat_lon_from_mapillary_record(data: dict) -> tuple[float, float] | None:
    """
    Read image coordinates from a Mapillary image record.

    Prefers ``computed_geometry`` over raw ``geometry``.

    :param data: Image object from the Mapillary API.
    :return: ``(latitude, longitude)`` or None if no valid Point geometry.

    Example::
        In: lat_lon_from_mapillary_record({"geometry": {"type": "Point", "coordinates": [19.94, 50.06]}})
        Out: (50.06, 19.94)
    """
    for key in ("computed_geometry", "geometry"):
        latitude_longitude = lat_lon_from_geometry(data.get(key))
        if latitude_longitude is not None:
            return latitude_longitude
    return None


def lat_lon_from_geometry(geometry: dict | None) -> tuple[float, float] | None:
    """
    Parse GeoJSON Point coordinates from a Mapillary ``geometry`` field.

    Coordinates are ``[longitude, latitude]`` per GeoJSON.

    :param geometry: The ``geometry`` object from a Mapillary image record, or None.
    :return: ``(latitude, longitude)`` or None if missing or invalid.

    Example::
        In: lat_lon_from_geometry({"type": "Point", "coordinates": [19.94, 50.06]})
        Out: (50.06, 19.94)
    """
    if not geometry or geometry.get("type") != "Point":
        return None
    coords = geometry.get("coordinates")
    if not isinstance(coords, list | tuple) or len(coords) < 2:
        return None
    try:
        lon, lat = float(coords[0]), float(coords[1])
    except (TypeError, ValueError):
        return None
    # API is GeoJSON order; callers use (lat, lon)
    return lat, lon
