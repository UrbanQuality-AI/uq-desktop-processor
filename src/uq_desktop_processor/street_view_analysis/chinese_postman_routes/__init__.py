"""
Road-network coverage routes (Chinese postman / Eulerian circuits) split on a grid;
export to GPX in WGS84.
"""

from .generate import generate_clean_routes
from .result import EulerRoutesResult

__all__ = ["EulerRoutesResult", "generate_clean_routes"]
