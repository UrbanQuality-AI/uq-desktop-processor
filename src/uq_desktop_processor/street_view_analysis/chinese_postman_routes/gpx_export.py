"""
Build GPX documents from WGS84 polylines and write them to disk.
"""

from pathlib import Path

import gpxpy.gpx


def build_gpx(polylines_wgs84: list[list[tuple[float, float]]], *, sector_name: str) -> gpxpy.gpx.GPX:
    """
    Create one GPX with one track per polyline; split tracks are named with a part suffix.

    :param polylines_wgs84: Each inner list is ``(longitude, latitude)`` vertices.
    :param sector_name: Base name for track metadata.
    :return: In-memory GPX object.

    Example::
        In: build_gpx([[(19.94, 50.06), (19.95, 50.06)]], sector_name="Sector 1")
        Out: GPX with 1 track, 1 segment, 2 points, track name "Sector 1"
    """
    gpx = gpxpy.gpx.GPX()
    for polyline_coordinates in polylines_wgs84:
        # One track + one segment per closed/open walk
        track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(track)
        track_segment = gpxpy.gpx.GPXTrackSegment()
        track.segments.append(track_segment)
        for longitude, latitude in polyline_coordinates:
            track_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=latitude, longitude=longitude))

    # Multiple subgraphs -> numbered track names
    for track_index, track in enumerate(gpx.tracks):
        if len(gpx.tracks) == 1:
            track.name = sector_name
        else:
            track.name = f"{sector_name} part {track_index + 1}"
    return gpx


def write_gpx(output_path: Path, gpx_document: gpxpy.gpx.GPX) -> None:
    """
    Serialize a GPX object to UTF-8 text.

    :param output_path: Destination ``.gpx`` path.
    :param gpx_document: Object from :func:`build_gpx`.

    Example::
        In: write_gpx(Path("routes/trasa_1.gpx"), gpx)
        Out: "routes/trasa_1.gpx" created on disk (UTF-8 GPX XML)
    """
    output_path.write_text(gpx_document.to_xml(), encoding="utf-8")
