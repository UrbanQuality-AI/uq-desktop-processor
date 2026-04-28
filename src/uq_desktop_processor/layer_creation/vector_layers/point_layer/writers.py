"""
Low-level writers for GeoJSON and GeoPandas outputs used by the exporter.
"""

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)


def _write_geojson(feature_collection: dict[str, Any], output_path: str) -> None:
    """
    Save a GeoJSON FeatureCollection to a .geojson or .json file.

    :param feature_collection: GeoJSON FeatureCollection dictionary.
    :param output_path: Destination path for the JSON/GeoJSON file.

    Example::
        In: _write_geojson(feature_collection, "out.geojson")
        Out: "out.geojson" created on disk
    """
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(feature_collection, file_handle, ensure_ascii=False, indent=2)

    feature_count = len(feature_collection.get("features", []))
    log.info("Saved GeoJSON: %s (object count: %s)", output_path, feature_count)


def _write_with_geopandas(feature_collection: dict[str, Any], output_path: str, layer: str | None = None) -> None:
    """
    Save a FeatureCollection to a binary geospatial format using GeoPandas.

    Supported formats (inferred from file extension):
      - .gpkg  (GeoPackage)
      - .shp   (ESRI Shapefile)
      - .parquet
      - other formats supported by GeoPandas drivers

    :param feature_collection: GeoJSON-like structure to be converted.
    :param output_path: Output file path.
    :param layer: Optional layer name for multi-layer formats (e.g. GeoPackage).

    Example::
        In: _write_with_geopandas(feature_collection, "out.gpkg", layer="scores")
        Out: "out.gpkg" created on disk with layer "scores"
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if not feature_collection["features"]:
        log.warning("No objects to save; feature collection is empty.")
        return

    rows: list[dict[str, Any]] = []

    # Convert each GeoJSON feature into a GeoPandas row.
    for feature_item in feature_collection["features"]:
        lon, lat = feature_item["geometry"]["coordinates"]
        properties = feature_item["properties"].copy()
        rows.append({**properties, "geometry": Point(lon, lat)})

    log.debug("Converted %s features to GeoDataFrame rows.", len(rows))

    geo_data_frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    file_extension = os.path.splitext(output_path.lower())[1]

    try:
        if file_extension == ".gpkg":
            # Use provided layer name or fall back to collection name.
            layer_name = layer or (feature_collection.get("name") or "data").replace("/", "_")
            geo_data_frame.to_file(output_path, layer=layer_name, driver="GPKG")  # type: ignore[assignment]

        elif file_extension == ".shp":
            geo_data_frame.to_file(output_path, driver="ESRI Shapefile")  # type: ignore[assignment]

        elif file_extension == ".parquet":
            geo_data_frame.to_parquet(output_path, index=False)  # type: ignore[assignment]

        else:
            # Default driver chosen by GeoPandas based on extension.
            geo_data_frame.to_file(output_path)  # type: ignore[assignment]

        log.info(
            "Saved %s: %s (object count: %s)",
            file_extension.upper(),
            output_path,
            len(geo_data_frame),
        )

    except Exception as error:
        log.error("GeoPandas save error: %s", error)
