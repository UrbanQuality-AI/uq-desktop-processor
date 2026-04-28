"""
Moves images into kept or rejected folders using a CLIP semantic accept/reject filter.
"""

import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from PIL import Image
from torch import nn

from uq_desktop_processor.evaluation.clip_common import encode_image, encode_texts, load_model, sigmoid_probability

from . import defaults
from .utils import _list_images, _move, _validate_dirs, resolve_rejected_folder_path
from .validators import _validate_prefilter_config

log = logging.getLogger(__name__)


@torch.no_grad()
def _build_filter_direction(model: nn.Module, device: str, filter_prompts: Mapping[str, Iterable[str]]) -> torch.Tensor:
    """
    Build a direction vector representing the semantic difference
    between positive and negative filtering prompts.

    :param model: CLIP model.
    :param device: Torch device.
    :param filter_prompts: Mapping with 'pos' and 'neg' prompt lists.
    :return: Normalized direction vector in embedding space.

    Example::
        In: _build_filter_direction(model, device, filter_prompts)
        Out: function result returned for provided inputs
    """
    log.debug("Building semantic filter direction vector from prompts...")
    positive_embeddings = encode_texts(filter_prompts["pos"], model, device).mean(0)
    negative_embeddings = encode_texts(filter_prompts["neg"], model, device).mean(0)

    direction_vector = positive_embeddings - negative_embeddings
    direction_norm = direction_vector.norm(p=2)

    return direction_vector / direction_norm if direction_norm > 0 else direction_vector


@torch.no_grad()
def prefilter_folder(
    image_folder: str = defaults.DEFAULT_IMAGE_FOLDER,
    device: str = defaults.DEFAULT_DEVICE,
    model_names: tuple[str, ...] = defaults.DEFAULT_MODEL_NAMES,
    beta_sigmoid: float = defaults.DEFAULT_BETA_SIGMOID,
    filter_threshold: float = defaults.DEFAULT_FILTER_THRESHOLD,
    filter_prompts: Mapping[str, Iterable[str]] = defaults.FILTER_PROMPTS,
    rejected_folder: str = defaults.DEFAULT_REJECTED_FOLDER,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Filter images in a folder based on similarity to prompt directions.
    Images below a threshold are moved to a rejected folder (unless dry_run=True).

    :param image_folder: Folder containing image files.
    :param device: PyTorch device string ("cpu" or "cuda").
    :param model_names: Models to attempt loading.
    :param beta_sigmoid: Sigmoid scaling factor.
    :param filter_threshold: The cutoff percentage [0, 100] for the filter.
    :param filter_prompts: A mapping containing 'pos' (positive) and 'neg' (negative) prompt lists.
    :param rejected_folder: Sub-folder name for rejected images.
    :param dry_run: If True, no files are moved.
    :return: Summary dictionary of kept/rejected images.

    Example::
        In: prefilter_folder(image_folder, device, model_names, beta_sigmoid, filter_threshold, filter_prompts, rejected_folder, dry_run)
        Out: function result returned for provided inputs
    """
    log.info(
        "Starting pre-filter in '%s' (Threshold: %s%%).",
        image_folder,
        filter_threshold,
    )

    if dry_run:
        log.warning("Dry run enabled: No files will actually be moved.")

    # 1. Validate inputs
    error_messages, warning_messages = _validate_prefilter_config(
        image_folder=image_folder,
        device=device,
        model_names=model_names,
        beta_sigmoid=beta_sigmoid,
        filter_threshold=filter_threshold,
        filter_prompts=filter_prompts,
        raise_on_error=True,
    )

    # 2. Prepare directories
    image_filenames = _list_images(image_folder)

    if not image_filenames:
        log.warning("No images found in '%s'. Aborting pre-filter.", image_folder)
    else:
        log.info("Found %s images to process.", len(image_filenames))

    absolute_rejected_folder = resolve_rejected_folder_path(image_folder, rejected_folder)
    _validate_dirs(image_folder, absolute_rejected_folder)

    # 3. Load Model and Build Filter Direction
    model, preprocess_function, loaded_model_name = load_model(model_names, device)
    filter_direction_vector = _build_filter_direction(model, device, filter_prompts)

    kept_images: list[dict[str, Any]] = []
    rejected_images: list[dict[str, Any]] = []

    # 4. Process Images
    for filename in image_filenames:
        image_path = os.path.join(image_folder, filename)
        try:
            with Image.open(image_path) as opened_image:
                image_features = encode_image(opened_image, preprocess_function, model, device)
        except Exception as open_exception:
            log.error("Failed to open/process image '%s': %s", filename, open_exception)
            rejected_images.append({"filename": filename, "reason": f"open_error: {open_exception}"})
            continue

        # Project image features onto direction vector
        delta_value = torch.dot(image_features, filter_direction_vector).item()

        # Convert to probability using sigmoid
        probability_value = float(sigmoid_probability(delta_value, beta_sigmoid))

        # Apply threshold filtering
        if probability_value < filter_threshold:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "REJECT: '%s' (%.1f%% < %s%%)",
                    filename,
                    probability_value,
                    filter_threshold,
                )

            rejected_images.append({"filename": filename, "filter_pct": round(probability_value, 1)})
            if not dry_run:
                # Keep file move side effect explicit and easy to spot.
                _move(image_path, absolute_rejected_folder)
        else:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "KEEP: '%s' (%.1f%% >= %s%%)",
                    filename,
                    probability_value,
                    filter_threshold,
                )

            kept_images.append({"filename": filename, "filter_pct": round(probability_value, 1)})

    log.info(
        "Pre-filtering complete. Kept: %s, Rejected: %s.",
        len(kept_images),
        len(rejected_images),
    )

    # 5. Return Summary
    return {
        "model_name": loaded_model_name,
        "image_folder": image_folder,
        "rejected_folder": absolute_rejected_folder,
        "beta_sigmoid": beta_sigmoid,
        "filter_threshold": filter_threshold,
        "kept": kept_images,
        "rejected": rejected_images,
        "summary": {
            "total": len(image_filenames),
            "kept": len(kept_images),
            "rejected": len(rejected_images),
        },
        "warnings": warning_messages,
        "dry_run": dry_run,
    }
