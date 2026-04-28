"""
Runs fine-tuned ViT scoring over a folder of images and aggregates JSON results.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from uq_desktop_processor.evaluation.evaluation_utils import make_evaluation_result

from . import defaults
from .utils import (
    _apply_calibration,
    _build_transform,
    _iter_image_paths,
    _load_calibrators,
    _load_checkpoint_and_model,
)

log = logging.getLogger(__name__)


def _resolve_paths(
    images_dir: str | Path,
    model_path: str | Path,
    calibrators_dir: str | Path | None,
) -> tuple[Path, Path, Path | None]:
    """
    Resolve and validate input paths.
    Raises FileNotFoundError or NotADirectoryError if paths are invalid.

    Example::
        In: _resolve_paths(images_dir, model_path, calibrators_dir)
        Out: function result returned for provided inputs
    """
    images_dir = Path(images_dir)
    model_path = Path(model_path)
    calibrators_dir_path = Path(calibrators_dir) if calibrators_dir else None

    log.debug(
        "Resolving paths: images='%s', model='%s', calibrators='%s'",
        images_dir,
        model_path,
        calibrators_dir_path,
    )

    # Validate images directory
    if not images_dir.exists():
        error_message = f"Images directory not found: {images_dir}"
        log.error(error_message)
        raise FileNotFoundError(error_message)
    if not images_dir.is_dir():
        error_message = f"Images path is not a directory: {images_dir}"
        log.error(error_message)
        raise NotADirectoryError(error_message)

    # Validate model file
    if not model_path.exists():
        error_message = f"Model file not found: {model_path}"
        log.error(error_message)
        raise FileNotFoundError(error_message)
    if not model_path.is_file():
        error_message = f"Model path must be a file: {model_path}"
        log.error(error_message)
        raise IsADirectoryError(error_message)

    # Validate calibrators directory
    if calibrators_dir_path is not None:
        if not calibrators_dir_path.exists():
            error_message = f"Calibrators directory not found: {calibrators_dir_path}"
            log.error(error_message)
            raise FileNotFoundError(error_message)
        if not calibrators_dir_path.is_dir():
            error_message = f"Calibrators path is not a directory: {calibrators_dir_path}"
            log.error(error_message)
            raise NotADirectoryError(error_message)

    return images_dir, model_path, calibrators_dir_path


def _get_calibrators(calibrators_dir: Path | None, category_order: Sequence[str]) -> dict:
    """
    Load per-category calibration metadata if provided; otherwise fall back.

    If ``calibrators_dir`` is None, calibration is effectively OFF.

    Example::
        In: _get_calibrators(calibrators_dir, category_order)
        Out: function result returned for provided inputs
    """
    if calibrators_dir is None:
        log.info("No calibrators directory provided. Calibration is OFF (fallback scaling).")
        return {category_name: {"type": "fallback"} for category_name in category_order}

    log.info("Loading calibrators from: %s", calibrators_dir)
    calibrators = _load_calibrators(calibrators_dir, category_order)
    loaded_calibrator_types = {category_name: spec["type"] for category_name, spec in calibrators.items()}
    log.info("Loaded calibrator types: %s", loaded_calibrator_types)
    return calibrators


def _predict_batch(model: torch.nn.Module, batch_tensors: list[torch.Tensor], device: torch.device) -> np.ndarray:
    """
    Run a batch of images through the fine-tuned model and return raw predictions.

    :param model: Fine-tuned model used for inference.
    :param batch_tensors: List of transformed image tensors.
    :param device: Device used for evaluation.
    :return: Numpy array of model outputs with shape (batch_size, num_outputs).

    Example::
        In: _predict_batch(model, batch_tensors, device)
        Out: function result returned for provided inputs
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "Predicting batch of size %s on device %s",
            len(batch_tensors),
            device,
        )

    batch_tensor = torch.stack(batch_tensors, dim=0).to(device)
    use_mixed_precision = torch.cuda.is_available()

    with torch.amp.autocast("cuda", enabled=use_mixed_precision):
        return model(batch_tensor).detach().cpu().numpy().astype(np.float32)


def _format_single_result(
    image_path: Path,
    raw_probs: np.ndarray,
    calibrators: dict,
    category_order: Sequence[str],
) -> dict:
    """
    Format one model output entry into a standardized result dictionary.

    :param image_path: Path to the evaluated image.
    :param raw_probs: Raw model output (one vector per image).
    :param calibrators: Calibration metadata.
    :return: Dictionary formatted for downstream tooling (similar to CLIP pipeline outputs).

    Example::
        In: _format_single_result(image_path, raw_probs, calibrators, category_order)
        Out: function result returned for provided inputs
    """
    calibrated_probs = _apply_calibration(raw_probs, calibrators, category_order)

    # Overall score = mean of calibrated per-category probabilities
    overall_score = float(np.mean([calibrated_probs[cat] for cat in category_order]))

    # CLIP pipeline expects "delta" values, but fine-tuned models do not provide them
    category_block = {
        category_name: {"probability_pct": float(calibrated_probs[category_name]), "delta": None}
        for category_name in category_order
    }

    return {
        "filename": image_path.name,
        "image_path": str(image_path),
        "overall_pct": overall_score,
        "categories": category_block,
    }


def _process_batch(
    model: torch.nn.Module,
    batch_tensors: list[torch.Tensor],
    batch_paths: list[Path],
    calibrators: dict,
    device: torch.device,
    category_order: Sequence[str],
) -> list[dict]:
    """
    Evaluate one batch and convert its outputs into result dictionaries.

    :param model: Fine-tuned model.
    :param batch_tensors: List of tensors in the current batch.
    :param batch_paths: Corresponding list of image paths.
    :param calibrators: Calibration metadata.
    :param device: Torch device.
    :return: List of results for each image in the batch.

    Example::
        In: _process_batch(model, batch_tensors, batch_paths, calibrators, device, category_order)
        Out: function result returned for provided inputs
    """
    raw_output = _predict_batch(model, batch_tensors, device)
    results = []

    for index in range(raw_output.shape[0]):
        # Keep path and prediction aligned by shared batch index.
        result = _format_single_result(batch_paths[index], raw_output[index], calibrators, category_order)
        results.append(result)

    return results


def _calculate_final_stats(results_images: list[dict]) -> float | None:
    """
    Compute the mean overall score across all evaluated images.

    :param results_images: List of result dictionaries.
    :return: Average score or None if no images were processed.

    Example::
        In: _calculate_final_stats(results_images)
        Out: function result returned for provided inputs
    """
    if not results_images:
        return None

    return float(np.mean([res["overall_pct"] for res in results_images]))


@torch.no_grad()
def evaluate_images_with_finetuned(
    images_dir: str | Path,
    model_path: str | Path,
    calibrators_dir: str | Path | None = None,
    model_name: str = "vit_base_patch14_dinov2.lvd142m",
    image_size: int = 224,
    batch_size: int = 32,
    torch_device: str = "auto",
    category_order: Sequence[str] = defaults.DEFAULT_CATEGORY_ORDER,
) -> dict:
    """
    Evaluate a set of images using a fine-tuned ViT-like model (loaded from checkpoint).

    This pipeline:
      - resolves paths and loads the model
      - builds transforms
      - optionally loads calibrators
      - iterates through images in batches
      - computes calibrated probabilities
      - formats results into a CLIP-compatible schema

    :return: Structured result dictionary similar to CLIP scoring outputs.

    Example::
        In: evaluate_images_with_finetuned(images_dir, model_path, calibrators_dir, model_name, image_size, batch_size, torch_device, category_order)
        Out: function result returned for provided inputs
    """
    log.info("Starting fine-tuned model evaluation pipeline.")

    # Resolve filesystem paths
    images_dir, model_path, calibrators_dir_path = _resolve_paths(
        images_dir=images_dir,
        model_path=model_path,
        calibrators_dir=calibrators_dir,
    )

    num_outputs = len(category_order)

    # Load model and preprocessing
    torch_device_name = (torch_device or "auto").lower().strip()
    if torch_device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("torch_device must be one of: auto, cpu, cuda")
    if torch_device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in this environment.")
    if torch_device_name == "cuda":
        device = torch.device("cuda")
    elif torch_device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading model weights from: %s (Model: %s)", model_path, model_name)
    log.info("Using device: %s", device)

    model = _load_checkpoint_and_model(model_path, model_name, image_size, device, num_outputs)
    transform_function = _build_transform(model_name, image_size)
    calibrators = _get_calibrators(calibrators_dir_path, category_order)

    # Gather image paths
    image_paths = _iter_image_paths(images_dir)
    if not image_paths:
        error_message = f"No images found in: {images_dir}"
        log.error(error_message)
        raise FileNotFoundError(error_message)

    log.info("Found %s images to evaluate.", len(image_paths))

    results_images: list[dict] = []
    skipped: list[str] = []
    warnings: list[str] = []

    batch_tensors: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    iterator = tqdm(image_paths, desc="Scoring images", dynamic_ncols=True)

    # Process images batch-by-batch
    for image_path in iterator:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as open_error:
            warning_message = f"Failed to open image '{image_path.name}': {open_error}"
            log.warning(warning_message)
            skipped.append(f"{image_path.name}: {open_error}")
            continue

        batch_tensors.append(transform_function(image))
        batch_paths.append(image_path)

        # Full batch > process it
        if len(batch_tensors) >= batch_size:
            batch_results = _process_batch(model, batch_tensors, batch_paths, calibrators, device, category_order)
            results_images.extend(batch_results)
            batch_tensors, batch_paths = [], []

    # Process leftovers
    if batch_tensors:
        batch_results = _process_batch(model, batch_tensors, batch_paths, calibrators, device, category_order)
        results_images.extend(batch_results)

    # Compute overall stats
    average_overall_score = _calculate_final_stats(results_images)

    log.info(
        "Evaluation complete. Average score: %s (Processed: %s, Skipped: %s)",
        average_overall_score,
        len(results_images),
        len(skipped),
    )

    return make_evaluation_result(
        model_name=f"{model_name} (fine-tuned)",
        order=category_order,
        weights=None,
        beta_sigmoid=None,
        images=results_images,
        average_overall_pct=average_overall_score,
        warnings=warnings,
        errors=None,
        skipped_images=skipped,
        extra={"calibration": "on" if calibrators_dir_path is not None else "off"},
    )
