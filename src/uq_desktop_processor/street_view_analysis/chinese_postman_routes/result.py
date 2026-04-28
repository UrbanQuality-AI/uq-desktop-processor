"""
Container for GPX paths and polylines returned by grid-sector route generation.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EulerRoutesResult:
    """
    Output directory, list of GPX paths, and WGS84 polylines (one sequence per route).

    ``polylines_wgs84`` has multiple entries when a sector contained several disconnected subgraphs.
    """

    output_dir: Path
    gpx_paths: tuple[Path, ...] = field(default_factory=tuple)
    polylines_wgs84: tuple[tuple[tuple[float, float], ...], ...] = field(default_factory=tuple)
