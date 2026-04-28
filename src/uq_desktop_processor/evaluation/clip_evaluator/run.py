"""
Batch-scores images with CLIP against configured semantic directions and writes JSON results.
"""

import logging
import os
from collections.abc import Mapping
from typing import Any

from uq_desktop_processor.evaluation.clip_common import ValidationError, load_model, prepare_directions, validate_config
from uq_desktop_processor.evaluation.evaluation_utils import make_evaluation_result

from . import defaults
from .scoring import score_image

log = logging.getLogger(__name__)


def _get_image_files(folder: str) -> list[str]:
    """
    Return a sorted list of image filenames in the specified folder.
    Only .png, .jpg, and .jpeg extensions are included.

    :param folder: Path to image folder.
    :return: Sorted list of filenames.

    Example::
        In: _get_image_files(folder)
        Out: function result returned for provided inputs
    """
    files = sorted(filename for filename in os.listdir(folder) if filename.lower().endswith((".png", ".jpg", ".jpeg")))
    log.debug("Scanned folder '%s': found %s valid image files.", folder, len(files))
    return files


def _format_single_result(
    filename: str, image_path: str, raw_result: dict[str, Any], order: tuple[str, ...]
) -> dict[str, Any]:
    """
    Format raw scoring output for a single image into a clean result structure.

    :param filename: Image filename.
    :param raw_result: Raw scoring dictionary from score_image().
    :param order: Category order tuple.
    :return: Formatted result dictionary.

    Example::
        In: _format_single_result(filename, raw_result, order)
        Out: function result returned for provided inputs
    """
    return {
        "filename": filename,
        "image_path": image_path,
        "categories": {
            category_name: {
                "delta": float(f"{raw_result[category_name]['delta']:.6f}"),
                "probability_pct": float(f"{raw_result[category_name]['probability']:.1f}"),
            }
            for category_name in order
        },
        "overall_pct": float(f"{raw_result['overall']:.1f}"),
    }


def _create_error_response(
    validation_exception: ValidationError, order: tuple, weights: Mapping, beta_sigmoid_value: float
) -> dict[str, Any]:
    """
    Build a consistent response dictionary when validation fails.

    :param validation_exception: ValidationError instance.
    :param order: Category order.
    :param weights: Weight mapping.
    :param beta_sigmoid_value: Sigmoid beta value.
    :return: Error response structure.

    Example::
        In: _create_error_response(validation_exception, order, weights, beta_sigmoid_value)
        Out: function result returned for provided inputs
    """
    return {
        "model_name": None,
        "order": order,
        "weights": dict(weights),
        "beta_sigmoid": beta_sigmoid_value,
        "warnings": list(validation_exception.warnings),
        "images": [],
        "average_overall_pct": None,
        "errors": list(validation_exception.errors),
    }


def evaluate_images_with_clip(
    image_folder: str = defaults.DEFAULT_IMAGE_FOLDER,
    device: str = defaults.DEFAULT_DEVICE,
    model_names: tuple[str, ...] = defaults.DEFAULT_MODEL_NAMES,
    beta_sigmoid: float = defaults.DEFAULT_BETA_SIGMOID,
    order: tuple[str, ...] = defaults.DEFAULT_ORDER,
    weights: Mapping[str, float] = defaults.DEFAULT_WEIGHTS,
    prompts: Mapping[str, dict] = defaults.DEFAULT_PROMPTS,
    *,
    raise_on_validation_error: bool = True,
) -> dict[str, Any]:
    """
    Evaluate all images in a folder using CLIP-based scoring.
    Performs validation, loads the model, prepares direction vectors,
    scores each image, and produces a final aggregated result.

    :param image_folder: Folder containing image files.
    :param device: PyTorch device string ("cpu" or "cuda").
    :param model_names: Models to attempt loading.
    :param beta_sigmoid: Sigmoid scaling factor.
    :param order: Category order tuple.
    :param weights: Weight mapping for categories.
    :param prompts: Prompt mapping for positive/negative directions.
    :param raise_on_validation_error: Whether to raise or return structured error output.
    :return: Full evaluation result dictionary.

    Example::
        In: evaluate_images_with_clip(image_folder, device, model_names, beta_sigmoid, order, weights, prompts)
        Out: function result returned for provided inputs
    """
    log.info("Starting CLIP evaluation pipeline.")

    try:
        # Validate configuration before processing
        error_messages, warning_messages = validate_config(
            image_folder=image_folder,
            device=device,
            model_names=model_names,
            beta_sigmoid=beta_sigmoid,
            order=order,
            weights=weights,
            prompts=prompts,
            raise_on_error=raise_on_validation_error,
        )
    except ValidationError as validation_exception:
        log.error("Validation failed with %s errors.", len(validation_exception.errors))
        # Return structured error result instead of raising
        if raise_on_validation_error:
            raise
        log.warning("Returning structured error response (raise_on_validation_error=False).")
        return make_evaluation_result(
            model_name=None,
            order=order,
            weights=dict(weights),
            beta_sigmoid=beta_sigmoid,
            images=[],
            average_overall_pct=None,
            warnings=list(validation_exception.warnings),
            errors=list(validation_exception.errors),
            skipped_images=[],
        )

    # Load input images
    image_filenames = _get_image_files(image_folder)

    if not image_filenames:
        log.warning("No images found in '%s'. Aborting evaluation.", image_folder)
    else:
        log.info("Found %s images to evaluate in '%s'.", len(image_filenames), image_folder)

    # Load CLIP model
    model, preprocess, loaded_model_name = load_model(model_names, device)

    # Compute semantic direction vectors for categories
    direction_vectors = prepare_directions(prompts, model, device)

    # Build shared configuration for scoring function
    configuration_for_scoring = {
        "device": device,
        "beta_sigmoid": beta_sigmoid,
        "order": order,
        "weights": weights,
    }

    formatted_image_results = []
    overall_scores = []

    # Evaluate each image
    for filename in image_filenames:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Scoring image: %s", filename)

        image_path = os.path.join(image_folder, filename)

        # Compute raw score
        raw_score_result = score_image(image_path, model, preprocess, direction_vectors, configuration_for_scoring)

        # Format final clean result for this image
        formatted_result = _format_single_result(filename, image_path, raw_score_result, order)
        formatted_image_results.append(formatted_result)

        # Collect overall score for averaging
        overall_scores.append(raw_score_result["overall"])

    # Compute mean overall score
    average_overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else None

    log.info(
        "Evaluation finished. Average overall score: %s (for %s images).",
        average_overall_score,
        len(overall_scores),
    )

    return make_evaluation_result(
        model_name=loaded_model_name,
        order=order,
        weights=dict(weights),
        beta_sigmoid=beta_sigmoid,
        images=formatted_image_results,
        average_overall_pct=(round(average_overall_score, 1) if average_overall_score is not None else None),
        warnings=warning_messages,
        errors=None,
        skipped_images=[],
    )
