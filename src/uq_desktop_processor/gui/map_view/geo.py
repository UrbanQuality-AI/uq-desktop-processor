"""
Geographic helpers: bounds fitting, coordinate transforms, and map viewport math.
"""

from pyproj import Geod

_GEO_WGS84 = Geod(ellps="WGS84")


def polyline_geodesic_length_m(poly: tuple[tuple[float, float], ...]) -> float:
    """Sum of geodesic segment lengths for a (lon, lat) polyline in WGS84."""
    if len(poly) < 2:
        return 0.0
    total = 0.0
    for i in range(len(poly) - 1):
        lon1, lat1 = poly[i]
        lon2, lat2 = poly[i + 1]
        _, _, dist_m = _GEO_WGS84.inv(lon1, lat1, lon2, lat2)
        total += abs(dist_m)
    return total
