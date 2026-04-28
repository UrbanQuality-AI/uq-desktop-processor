"""
Exports scoring results to GeoJSON or GeoPandas-backed spatial files.
"""

import logging
import os
from typing import Any

from .converters import _results_to_feature_collection
from .utils import _is_geopandas_available
from .writers import _write_geojson, _write_with_geopandas

log = logging.getLogger(__name__)


def export_point_layer(
    results: dict[str, Any], output_path: str, layer: str | None = None, categories: list[str] | None = None
) -> str:
    """
    Export scoring results to a geospatial file.

    The output format is inferred from the file extension:

      - .geojson / .json  -> plain GeoJSON, written with the standard json module
      - .gpkg / .shp / .parquet / others supported by GeoPandas -> written using GeoPandas

    :param results: Dictionary returned by the evaluation pipeline.
    :param output_path: Destination file path (extension determines format).
    :param layer: Optional layer name (for formats that support multiple layers).
    :param categories: Optional subset of categories to include (default uses
                        defaults.DEFAULT_CATEGORIES).
    :return: Absolute path to the saved file.

    Example::
        In: export_point_layer(results, "data/results/urban_quality_ai_output.geojson")
        Out: "C:/.../data/results/urban_quality_ai_output.geojson"
    """
    log.info("Preparing to export data to: %s", output_path)

    try:
        # Convert raw results into a structured FeatureCollection.
        feature_collection = _results_to_feature_collection(results, categories=categories)

        feature_count = len(feature_collection.get("features", []))
        log.debug("Converted results to FeatureCollection with %s features.", feature_count)

        absolute_output_path = os.path.abspath(output_path)
        file_extension = os.path.splitext(output_path.lower())[1]

        # JSON-based formats do not require GeoPandas.
        if file_extension in (".geojson", ".json"):
            _write_geojson(feature_collection, absolute_output_path)
        else:
            # Other formats require GeoPandas.
            if not _is_geopandas_available():
                msg = (
                    "GeoPandas is required to save to binary formats (GPKG, SHP, Parquet).\n"
                    "Install: pip install geopandas pyproj fiona shapely\n"
                    "Or use the .geojson extension."
                )
                log.error(msg)
                raise RuntimeError(msg)

            _write_with_geopandas(feature_collection, absolute_output_path, layer=layer)

        log.info("Export successful. File saved at: %s", absolute_output_path)
        return absolute_output_path

    except Exception as error:
        log.error("Failed to export layer to '%s': %s", output_path, error)
        raise
