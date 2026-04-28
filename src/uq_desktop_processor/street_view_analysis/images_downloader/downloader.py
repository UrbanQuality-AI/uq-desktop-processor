"""
Downloads street-view images for sampling points with concurrency and retries.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .api import query_mapillary_image
from .helpers import lat_lon_from_mapillary_record, mapillary_image_filename
from .image_processing import download_image, process_panorama, save_jpeg_with_exif

log = logging.getLogger(__name__)


def _handle_single_point(
    lat_lon: tuple[float, float],
    output_folder: str,
    mapillary_token: str,
    search_radius_m: float,
    fov_deg: float,
    output_hw: tuple[int, int],
    api_timeout_s: float,
    retries: int = 3,
) -> str:
    """
    Fetch and save a Mapillary image for one point, with retries on failure.

    :param lat_lon: ``(latitude, longitude)`` of the target location.
    :param output_folder: Directory for the saved JPEG.
    :param mapillary_token: OAuth token for the Mapillary API.
    :param search_radius_m: Search radius in meters around the point.
    :param fov_deg: Horizontal field of view in degrees for panorama cropping.
    :param output_hw: Output size as ``(height, width)``.
    :param api_timeout_s: Timeout for API and HTTP download requests.
    :param retries: Attempt count including the first try.
    :return: Status message (success or failure reason).

    Example::
        In: _handle_single_point((50.06, 19.94), "data/images", token, 150, 90, (512, 512), 10.0)
        Out: status string, e.g. "Success (50.06000, 19.94000): <id>.jpg" or "(50.06000, 19.94000) - no image found"
    """
    latitude, longitude = lat_lon
    for attempt_index in range(retries):
        try:
            # One full fetch+save path; outer loop handles retries
            return _try_fetch_and_save_image(
                latitude, longitude, output_folder, mapillary_token, search_radius_m, fov_deg, output_hw, api_timeout_s
            )
        except Exception as error:
            if attempt_index < retries - 1:
                log.warning(
                    "(%.5f, %.5f) Attempt %s failed: %s. Retrying...",
                    latitude,
                    longitude,
                    attempt_index + 1,
                    error,
                )
                time.sleep(1)
            else:
                return f"({latitude:.5f}, {longitude:.5f}) Critical error: {error}"

    # e.g. retries == 0
    return f"({latitude:.5f}, {longitude:.5f}) - failed after {retries} attempts"


def _try_fetch_and_save_image(
    lat: float,
    lon: float,
    output_folder: str,
    mapillary_token: str,
    search_radius_m: float,
    fov_deg: float,
    output_hw: tuple[int, int],
    api_timeout_s: float,
) -> str:
    """
    Query Mapillary, download the image, optionally unwrap panorama, write JPEG with EXIF.

    :param lat: Latitude of the target point.
    :param lon: Longitude of the target point.
    :param output_folder: Directory for the output file.
    :param mapillary_token: OAuth token for the Mapillary API.
    :param search_radius_m: Search radius in meters.
    :param fov_deg: Field of view in degrees for panorama cropping.
    :param output_hw: Output resolution ``(height, width)``.
    :param api_timeout_s: Timeout for API and download.
    :return: Status message indicating success or failure.

    Example::
        In: _try_fetch_and_save_image(50.06, 19.94, "data/images", token, 150, 90, (512, 512), 10.0)
        Out: status string, e.g. "Success (...)" or "(...) - missing image URL in API response"
    """
    # Nearest image metadata in bbox (or None)
    data = query_mapillary_image(lat, lon, mapillary_token, search_radius_m, api_timeout_s)
    if not data:
        return f"({lat:.5f}, {lon:.5f}) - no image found"

    image_url = data.get("thumb_1024_url")
    if not image_url:
        return f"({lat:.5f}, {lon:.5f}) - missing image URL in API response"

    image_array = download_image(image_url, api_timeout_s)
    if image_array is None:
        return f"({lat:.5f}, {lon:.5f}) - download failed"

    # Panoramas: equirectangular -> perspective crop
    if data.get("is_pano", False):
        image_array = process_panorama(image_array, fov_deg, output_hw)

    image_id = data.get("id")
    if not image_id:
        return f"({lat:.5f}, {lon:.5f}) - missing image id in API response"

    image_coordinates = lat_lon_from_mapillary_record(data)
    # Fall back to query point if the record has no usable geometry
    image_latitude, image_longitude = image_coordinates if image_coordinates is not None else (lat, lon)

    filename = mapillary_image_filename(str(image_id))
    path = os.path.join(output_folder, filename)
    captured_at = data.get("captured_at")
    save_jpeg_with_exif(
        image_array,
        path,
        lat=image_latitude,
        lon=image_longitude,
        captured_at_ms=captured_at,
    )

    return f"Success ({lat:.5f}, {lon:.5f}): {filename}"


def download_mapillary_images(
    points: list[tuple[float, float]],
    output_folder: str,
    mapillary_token: str,
    *,
    search_radius_m: float = 150,
    fov_deg: float = 90,
    output_hw: tuple[int, int] = (512, 512),
    max_workers: int = 20,
    api_timeout_s: float = 10.0,
) -> dict[str, Any]:
    """
    Download Mapillary images for many points using a thread pool.

    :param points: ``(latitude, longitude)`` tuples to query.
    :param output_folder: Folder for saved JPEGs (created if missing).
    :param mapillary_token: OAuth token for the Mapillary API.
    :param search_radius_m: Maximum search radius in meters.
    :param fov_deg: Field of view for panoramic crops.
    :param output_hw: Output image ``(height, width)``.
    :param max_workers: Parallel worker threads.
    :param api_timeout_s: Per-request timeout in seconds.
    :return: Summary with counts, timing, failure messages, and resolved output path.

    Example::
        In: download_mapillary_images([(50.06, 19.94)], "data/images", token)["downloaded"]
        Out: summary dict, e.g. {"point_count": 1, "downloaded": 1, "failed_messages": [], ...}
    """
    os.makedirs(output_folder, exist_ok=True)
    out_resolved = str(Path(output_folder).resolve())

    total_point_count = len(points)
    log.info(
        "Start downloading images for %s points using %s threads...",
        total_point_count,
        max_workers,
    )

    start_time = time.time()
    status_messages: list[str] = []

    # I/O-bound: threads overlap API + download latency
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for status_message in pool.map(
            lambda point_coordinates: _handle_single_point(
                point_coordinates, output_folder, mapillary_token, search_radius_m, fov_deg, output_hw, api_timeout_s
            ),
            points,
        ):
            status_messages.append(status_message)
            # Quiet expected outcomes; surface real failures
            if status_message.startswith("Success") or "no image found" in status_message:
                log.debug(status_message)
            else:
                log.warning(status_message)

    elapsed_seconds = time.time() - start_time
    # Success is encoded in the message prefix from _handle_single_point.
    downloaded_image_count = sum(1 for status_message in status_messages if status_message.startswith("Success"))
    # Keep all non-success messages for caller inspection/reporting.
    failure_messages = [
        status_message for status_message in status_messages if not status_message.startswith("Success")
    ]
    if total_point_count > 0:
        log.info(
            "Download complete in %.1f s (avg %.2f s/point); saved %s / %s images.",
            elapsed_seconds,
            elapsed_seconds / total_point_count,
            downloaded_image_count,
            total_point_count,
        )
    else:
        log.info("Download complete: no points to process.")

    images_per_second = (downloaded_image_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    return {
        "point_count": total_point_count,
        "downloaded": downloaded_image_count,
        "failed_messages": failure_messages,
        "elapsed_s": elapsed_seconds,
        "output_folder": out_resolved,
        "images_per_second": images_per_second,
    }
