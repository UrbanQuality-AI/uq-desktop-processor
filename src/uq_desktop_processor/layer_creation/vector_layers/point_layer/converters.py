"""
Converts evaluation result dicts into GeoJSON-like feature collections for export.
"""

import logging
from typing import Any

from . import defaults
from .utils import parse_lat_lon

log = logging.getLogger(__name__)

DEFAULT_CATEGORIES = defaults.DEFAULT_CATEGORIES


def _get_value_from_categories(categories_data: dict[str, Any], key: str) -> dict[str, Any]:
    """
    Safely retrieve a category sub-dictionary from the categories mapping.

    :param categories_data: Mapping of model category keys to dictionaries.
    :param key: Key to retrieve from the mapping.
    :return: The corresponding dict if present and of correct type, otherwise {}.

    Example::
        In: _get_value_from_categories({"wealthier": {"delta": 0.3}}, "wealthier")
        Out: {"delta": 0.3}
    """
    value = categories_data.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _resolve_weight(global_weights: dict[str, Any], key: str, alias: str) -> float | None:
    """
    Resolve a category weight using either the canonical category name or its alias.

    :param global_weights: Mapping of category names or aliases to weight values.
    :param key: Canonical category name (e.g. "greenery").
    :param alias: Alternate key used in the model output, if different.
    :return: The resolved weight, or None if no matching key was found.

    Example::
        In: _resolve_weight({"wealthier": 0.4}, "wealthier", "wealth")
        Out: 0.4
    """
    if not global_weights:
        return None
    canonical_weight = global_weights.get(key)
    if canonical_weight is not None:
        return canonical_weight
    return global_weights.get(alias)


def _build_feature(
    image_data: dict[str, Any],
    global_metadata: dict[str, Any],
    categories: list[str],
    aliases: dict[str, str],
) -> dict[str, Any]:
    """
    Convert a single scoring result into a GeoJSON Feature.

    This function:
      - extracts lat/lon coordinates from the filename,
      - merges global metadata (model name, beta, weights),
      - attaches per-category probability and delta values,
      - builds a valid GeoJSON Point feature.

    :param image_data: Single entry from results["images"] with per-image metrics.
    :param global_metadata: Metadata shared across all images
                            (model_name, beta_sigmoid, weights, etc.).
    :param categories: List of logical category names to export.
    :param aliases: Mapping from logical category names to model keys.
    :return: A GeoJSON Feature dictionary.

    Example::
        In: _build_feature(image_data, global_metadata, ["wealthier"], {"wealthier": "wealthier"})
        Out: {"type": "Feature", "geometry": {"type": "Point", ...}, "properties": {...}}
    """
    # Location comes strictly from image metadata (EXIF GPS).
    filename = image_data.get("filename", "")
    image_path = image_data.get("image_path")
    if not image_path:
        raise ValueError("Missing image_path; cannot read EXIF GPS metadata.")
    lat, lon = parse_lat_lon(image_path)

    # Per-category data provided by the scoring pipeline
    categories_data = image_data.get("categories", {}) or {}

    # Base properties (apply to the entire image/result)
    properties = {
        "filename": filename,
        "image_path": image_path,
        "overall_pct": image_data.get("overall_pct"),
        "model_name": global_metadata.get("model_name"),
        "beta_sigmoid": global_metadata.get("beta_sigmoid"),
    }

    global_weights = global_metadata.get("weights") or {}

    # Add per-category values.
    for category_key in categories:
        category_values = _get_value_from_categories(categories_data, category_key)

        # Property names should not contain spaces
        clean_key = category_key.replace(" ", "_")

        # Probability and delta per category
        properties[f"{clean_key}_prob_pct"] = category_values.get("probability_pct")
        properties[f"{clean_key}_delta"] = category_values.get("delta")

        # Optional weight information for this category.
        category_alias = aliases.get(category_key, category_key)
        weight = _resolve_weight(global_weights, category_key, category_alias)
        if weight is not None:
            properties[f"weights_{clean_key}"] = weight

    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": properties,
    }
    return feature


def _results_to_feature_collection(
    results: dict[str, Any],
    categories: list[str] | None = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Convert the full evaluation output into a GeoJSON FeatureCollection.

    :param results: Dictionary returned by the scoring pipeline.
    :param categories: Optional list of categories to export. If None, uses
                        defaults.DEFAULT_CATEGORIES.
    :param aliases: Optional mapping from category names to model keys.
                    If None, uses defaults.DEFAULT_ALIASES.
    :return: A GeoJSON FeatureCollection dictionary including extra metadata
             under the "x_meta" key.

    Example::
        In: _results_to_feature_collection({"images": [], "model_name": "clip"})
        Out: {"type": "FeatureCollection", "features": [], "x_meta": {...}}
    """
    target_categories = categories if categories is not None else DEFAULT_CATEGORIES
    target_aliases = aliases if aliases is not None else getattr(defaults, "DEFAULT_ALIASES", {})

    images_list = results.get("images", []) or []

    log.debug("Converting %s results to GeoJSON features.", len(images_list))

    # Data shared between all features; also emitted in x_meta.
    global_metadata = {
        "model_name": results.get("model_name"),
        "beta_sigmoid": results.get("beta_sigmoid"),
        "weights": results.get("weights"),
        "order": results.get("order"),
        "average_overall_pct": results.get("average_overall_pct"),
    }

    features_list: list[dict[str, Any]] = []

    for image_entry in images_list:
        try:
            feature = _build_feature(
                image_entry,
                global_metadata,
                target_categories,
                target_aliases,
            )
            features_list.append(feature)
        except Exception as exception:
            # Do not fail the entire export on a single bad filename or record.
            filename = image_entry.get("filename", "unknown")
            log.warning("Skipping '%s': %s", filename, exception)

    if not features_list:
        log.warning("No features were generated during conversion (all input images failed or list was empty).")

    feature_collection = {
        "type": "FeatureCollection",
        "name": results.get("model_name") or "model_points",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features_list,
        "x_meta": {
            **global_metadata,
            "exported_categories": target_categories,
            "skipped_images": results.get("skipped_images", []),
            "warnings": results.get("warnings", []),
            "errors": results.get("errors"),
        },
    }
    return feature_collection
