"""
Mapillary Graph API client: image search by bbox and metadata retrieval.
"""

import logging

import numpy as np
import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

log = logging.getLogger(__name__)

MAPILLARY_API_URL = "https://graph.mapillary.com/images"


def _calculate_bounding_box(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """
    Build a geographic bounding box from a center point and radius in meters.

    :param lat: Latitude of the center.
    :param lon: Longitude of the center.
    :param radius_m: Half-edge size in meters (approximate).
    :return: ``(min_lon, min_lat, max_lon, max_lat)``.

    Example::
        In: _calculate_bounding_box(50.05, 19.94, 100)
        Out: tuple (min_lon, min_lat, max_lon, max_lat) around the input point
    """
    # ~111.32 km per degree latitude; longitude spacing scales with cos(lat)
    delta_lat = radius_m / 111320
    delta_lon = delta_lat / np.cos(np.radians(lat))
    return lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat


def _build_mapillary_query_params(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> dict:
    """
    Build query parameters for a single-image Mapillary ``/images`` request.

    :param min_lon: Bounding box minimum longitude.
    :param min_lat: Bounding box minimum latitude.
    :param max_lon: Bounding box maximum longitude.
    :param max_lat: Bounding box maximum latitude.
    :return: Query parameter mapping for ``requests.get``.

    Example::
        In: _build_mapillary_query_params(19.9, 50.0, 20.0, 50.1)["limit"]
        Out: 1
    """
    return {
        "fields": "id,thumb_1024_url,is_pano,captured_at,geometry,computed_geometry",
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "limit": 1,
    }


def _is_radius_timeout_error(response: requests.Response) -> bool:
    """
    Detect Mapillary error 3404014 (bbox / radius too large for the request).

    :param response: HTTP response with error JSON body.
    :return: True if the error subcode indicates radius timeout.
    """
    try:
        error_info = response.json().get("error")
        # Mapillary: bbox too large / request timed out
        return error_info and error_info.get("error_subcode") == 3404014

    except ValueError:
        return False


def _fetch_mapillary_data(params: dict, headers: dict, timeout: float) -> dict:
    """
    GET the Mapillary images endpoint and return parsed JSON.

    :param params: Query string parameters.
    :param headers: Request headers (e.g. OAuth).
    :param timeout: Socket read timeout in seconds.
    :return: Parsed JSON body.
    :raises HTTPError: If the status code indicates failure.
    """
    log.debug("Requesting Mapillary API with params: %s", params)
    response = requests.get(MAPILLARY_API_URL, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _get_single_image_from_response(data: dict) -> dict | None:
    """
    Return the first image record from a Mapillary list response.

    :param data: Parsed JSON from ``/images``.
    :return: One image dict, or None if the list is empty.

    Example::
        In: _get_single_image_from_response({"data": [{"id": "abc"}]})
        Out: {"id": "abc"}
    """
    images = data.get("data", [])
    if not images:
        log.debug("No images found in API response.")
    return images[0] if images else None


def query_mapillary_image(
    lat: float, lon: float, token: str, radius: float, timeout: float, retries: int = 2
) -> dict | None:
    """
    Find one image near ``(lat, lon)``, shrinking the radius or retrying on certain errors.

    :param lat: Search latitude.
    :param lon: Search longitude.
    :param token: OAuth access token.
    :param radius: Initial search radius in meters.
    :param timeout: HTTP timeout in seconds.
    :param retries: Extra attempts after the first request (radius shrink and HTTP retries).
    :return: Image metadata dict, or None if no image or unrecoverable error.

    Example::
        In: query_mapillary_image(50.06, 19.94, token="...", radius=150, timeout=10.0)
        Out: {"id": "...", ...} | None
    """
    headers = {"Authorization": f"OAuth {token}"}

    # First attempt + retries (radius shrink, 5xx, or network)
    for attempt_index in range(retries + 1):
        min_lon, min_lat, max_lon, max_lat = _calculate_bounding_box(lat, lon, radius)
        params = _build_mapillary_query_params(min_lon, min_lat, max_lon, max_lat)

        try:
            data = _fetch_mapillary_data(params, headers, timeout)
            result = _get_single_image_from_response(data)
            if result:
                log.debug(
                    "Found image for (%s, %s) at attempt %s.",
                    lat,
                    lon,
                    attempt_index + 1,
                )
            return result

        except HTTPError as http_error:
            # Shrink search area and retry
            if http_error.response.status_code == 400 and _is_radius_timeout_error(http_error.response):
                old_radius = radius
                radius = max(50, radius // 2)
                log.warning(
                    "Radius %sm too large for (%s, %s). Reducing to %sm and retrying.",
                    old_radius,
                    lat,
                    lon,
                    radius,
                )
                continue

            # Transient server errors: retry same bbox
            if 500 <= http_error.response.status_code < 600:
                log.warning(
                    "Mapillary server error %s for (%s, %s). Retrying...",
                    http_error.response.status_code,
                    lat,
                    lon,
                )
                continue

            # Auth, bad request, etc.: do not loop forever
            log.error("HTTP error querying Mapillary for (%s, %s): %s", lat, lon, http_error)
            break

        except (Timeout, ConnectionError) as network_error:
            log.warning(
                "Network error querying Mapillary for (%s, %s): %s. Retrying...",
                lat,
                lon,
                network_error,
            )
            continue

    log.info(
        "Failed to retrieve image for (%s, %s) after %s attempts.",
        lat,
        lon,
        retries + 1,
    )
    return None
