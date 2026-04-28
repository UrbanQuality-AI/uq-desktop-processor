"""
Loads checkpoints, image transforms, and calibration for fine-tuned evaluation.
"""

import json
import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ViTMultiHead

log = logging.getLogger(__name__)

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except Exception as import_error:
    # Provide a clear installation message if timm is missing
    raise RuntimeError("Install timm: pip install timm") from import_error


def _build_transform(model_name: str, image_size: int) -> Callable[[Any], torch.Tensor]:
    """
    Build the preprocessing transform used for the fine-tuned model.

    Uses TIMM utilities to resolve the correct mean/std and transform pipeline
    based on the model's pretrained configuration.

    :param model_name: Name of the backbone model.
    :param image_size: Input image size expected by the model.
    :return: A torchvision-compatible transform function.

    Example::
        In: _build_transform(model_name, image_size)
        Out: function result returned for provided inputs
    """
    log.debug("Building transform for model '%s' (size: %s)", model_name, image_size)
    pretrained_config = timm.get_pretrained_cfg(model_name)
    pretrained_cfg_dict: dict[str, Any] = pretrained_config.to_dict() if pretrained_config is not None else {}
    base_config = resolve_data_config(pretrained_cfg=pretrained_cfg_dict)

    transform = create_transform(
        input_size=(3, image_size, image_size),
        is_training=False,
        mean=base_config.get("mean"),
        std=base_config.get("std"),
    )
    return transform


def _load_checkpoint_and_model(
    checkpoint_path: Path,
    model_name: str,
    image_size: int,
    device: torch.device,
    num_outputs: int,
) -> nn.Module:
    """
    Load a fine-tuned ViTMultiHead model and restore weights from checkpoint.

    :param checkpoint_path: Path to a .pt checkpoint containing the saved state_dict.
    :param model_name: Name of the TIMM backbone.
    :param image_size: Input resolution.
    :param device: Torch device for model instantiation.
    :return: Loaded and ready-to-evaluate model instance.

    Example::
        In: _load_checkpoint_and_model(checkpoint_path, model_name, image_size, device, num_outputs)
        Out: function result returned for provided inputs
    """
    log.debug("Loading checkpoint from: %s", checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as error:
        log.error("Failed to load checkpoint file: %s", error)
        raise

    # Create the classifier head with the correct shape
    model = ViTMultiHead(model_name, num_outputs=num_outputs, image_size=image_size).to(device)

    def _resize_backbone_pos_embed_if_needed(
            state_dict_in: dict[str, Any], model_in: nn.Module
    ) -> dict[str, Any]:
        """
        Resize checkpoint positional embeddings to the current model's image size if needed.

        This function handles the spatial interpolation of ViT positional embeddings
        when the inference resolution differs from the training resolution.

        :param state_dict_in: The source state dictionary from the loaded checkpoint.
        :param model_in: The target model instance to match the shapes against.
        :return: A state dictionary with adjusted (resized) positional embeddings.

        Example::
            In: _resize_backbone_pos_embed_if_needed(state_dict_in, model_in)
            Out: function result returned for provided inputs
        """
        pos_key = "backbone.pos_embed"

        # Check if the positional embedding key exists in both the checkpoint and the model
        if pos_key not in state_dict_in:
            return state_dict_in

        model_state = model_in.state_dict()
        if pos_key not in model_state:
            return state_dict_in

        checkpoint_pos_embed = state_dict_in[pos_key]
        target_pos_embed = model_state[pos_key]

        # Ensure we are dealing with tensors
        if not isinstance(checkpoint_pos_embed, torch.Tensor) or not isinstance(target_pos_embed, torch.Tensor):
            return state_dict_in

        # If shapes already match, no interpolation is required
        if checkpoint_pos_embed.shape == target_pos_embed.shape:
            return state_dict_in

        # Validate dimensions before proceeding
        if checkpoint_pos_embed.ndim != 3 or target_pos_embed.ndim != 3:
            log.warning("Unexpected pos_embed shape. Falling back to model defaults for pos_embed.")
            state_dict_copy = dict(state_dict_in)
            state_dict_copy.pop(pos_key, None)
            return state_dict_copy

        checkpoint_tokens = checkpoint_pos_embed.shape[1]
        target_tokens = target_pos_embed.shape[1]
        embed_dim = checkpoint_pos_embed.shape[2]

        # Check if the embedding vector size (last dimension) matches
        if embed_dim != target_pos_embed.shape[2]:
            log.warning("pos_embed embedding dim mismatch. Falling back to model defaults.")
            state_dict_copy = dict(state_dict_in)
            state_dict_copy.pop(pos_key, None)
            return state_dict_copy

        # Determine the number of special tokens (e.g., [CLS]) to separate them from grid tokens
        backbone_any = cast(Any, model_in.backbone)
        num_prefix_tokens = int(getattr(backbone_any, "num_prefix_tokens", 1))

        source_grid_tokens = checkpoint_tokens - num_prefix_tokens
        target_grid_tokens = target_tokens - num_prefix_tokens
        source_size = int(math.sqrt(source_grid_tokens))
        target_size = int(math.sqrt(target_grid_tokens))

        # Ensure the spatial tokens form a perfect square
        if (
                source_size * source_size != source_grid_tokens
                or target_size * target_size != target_grid_tokens
                or source_grid_tokens <= 0
                or target_grid_tokens <= 0
        ):
            log.warning("Cannot infer square token grid. Falling back to model defaults.")
            state_dict_copy = dict(state_dict_in)
            state_dict_copy.pop(pos_key, None)
            return state_dict_copy

        # Separate prefix tokens from patch tokens
        cls_tokens = checkpoint_pos_embed[:, :num_prefix_tokens, :]
        patch_tokens = checkpoint_pos_embed[:, num_prefix_tokens:, :]

        # Reshape patch tokens to a spatial grid for bicubic interpolation
        patch_tokens = patch_tokens.reshape(1, source_size, source_size, embed_dim).permute(0, 3, 1, 2)
        patch_tokens = F.interpolate(patch_tokens, size=(target_size, target_size), mode="bicubic", align_corners=False)

        # Flatten back to the sequence format
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, target_size * target_size, embed_dim)

        # Combine prefix tokens with the newly resized patch tokens
        resized_pos_embed = torch.cat([cls_tokens, patch_tokens], dim=1).to(dtype=target_pos_embed.dtype)

        state_dict_copy = dict(state_dict_in)
        state_dict_copy[pos_key] = resized_pos_embed

        log.info(
            "Resized checkpoint pos_embed from %s to %s to match GUI image size.",
            tuple(checkpoint_pos_embed.shape),
            tuple(target_pos_embed.shape),
        )
        return state_dict_copy

    # Load weights and configure for inference
    try:
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            log.info("Detected dictionary checkpoint format (key 'model' found).")
        else:
            state_dict = checkpoint
            log.info("Detected direct state_dict checkpoint format.")

        typed_state_dict = cast(dict[str, Any], state_dict)
        adjusted_state_dict = _resize_backbone_pos_embed_if_needed(typed_state_dict, model)
        missing_keys, unexpected_keys = model.load_state_dict(adjusted_state_dict, strict=False)

        filtered_missing_keys = [key for key in missing_keys if key != "backbone.pos_embed"]
        if filtered_missing_keys or unexpected_keys:
            log.warning(
                "Checkpoint loaded with missing/unexpected keys. missing=%s unexpected=%s",
                filtered_missing_keys,
                unexpected_keys,
            )

    except Exception as error:
        log.error("Error loading state_dict: %s", error)
        raise

    model.eval()
    torch.set_grad_enabled(False)

    log.info("Model loaded and ready for inference.")
    return model


def _load_calibrators(calibrators_dir: Path, category_order: Sequence[str]) -> dict[str, dict[str, Any]]:
    """
    Load isotonic calibrators and their metadata from the specified directory.

    This function attempts to load specific joblib models for each category
    based on the mapping found in 'calibrators_meta.json'.

    :param calibrators_dir: Path to the directory containing .joblib files and metadata.
    :param category_order: Sequence of category names expected by the model.
    :return: A dictionary mapping each category to its calibrator model or a fallback.

    Example::
        In: _load_calibrators(calibrators_dir, category_order)
        Out: function result returned for provided inputs
    """
    meta_path = calibrators_dir / "calibrators_meta.json"
    meta: dict[str, dict] = {}

    # Attempt to load metadata that maps categories to specific files
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as meta_file:
                meta = json.load(meta_file)
        except Exception as error:
            log.warning("Failed to load calibration metadata: %s", error)

    calibrators: dict[str, dict[str, Any]] = {}

    for category_name in category_order:
        entry = meta.get(category_name)

        # Retrieve filename from metadata or construct a default based on the category name
        file_name_any = entry.get("file") if entry else None
        file_name = (
            str(file_name_any)
            if isinstance(file_name_any, str) and file_name_any.strip()
            else f'calibrator_{category_name.replace(" ", "_")}.joblib'
        )
        calibrator_path = calibrators_dir / file_name

        if calibrator_path.exists():
            try:
                # Load the isotonic regression model using joblib
                calibrators[category_name] = {
                    "type": "isotonic",
                    "model": cast(Any, joblib.load(calibrator_path)),
                }
                log.debug("Loaded isotonic calibrator for %s", category_name)
                continue
            except Exception as error:
                log.warning("Could not load calibrator file %s: %s", file_name, error)

        # Set fallback if the file is missing or corrupted
        calibrators[category_name] = {"type": "fallback"}

    return calibrators


def _apply_calibration(
    raw_score_vector: np.ndarray,
    calibrators: dict[str, dict[str, Any]],
    category_order: Sequence[str],
) -> dict[str, float]:
    """
    Apply category-specific calibration to raw model outputs.

    Supported calibration types:
      • pchip – monotonic cubic Hermite spline
      • isotonic – isotonic regression model
      • minmax – simple linear normalization
      • fallback – heuristic scaling

    :param raw_score_vector: Raw predictions for all categories (vector of length 6).
    :param calibrators: Per-category calibrator configuration.
    :return: Mapping category → calibrated probability in [0, 100].

    Example::
        In: _apply_calibration(raw_score_vector, calibrators, category_order)
        Out: function result returned for provided inputs
    """
    calibrated_probabilities: dict[str, float] = {}

    for index, category_name in enumerate(category_order):
        calibrator_spec = calibrators.get(category_name, {"type": "fallback"})
        raw_value = float(raw_score_vector[index])

        # PCHIP interpolation
        if calibrator_spec.get("type") == "pchip" and "model" in calibrator_spec:
            pchip_model = calibrator_spec["model"]
            calibrated_value = float(pchip_model(raw_value))
            calibrated_probabilities[category_name] = float(np.clip(calibrated_value, 0.0, 100.0))

        # Isotonic regression
        elif calibrator_spec.get("type") == "isotonic" and "model" in calibrator_spec:
            calibrated_probabilities[category_name] = float(calibrator_spec["model"].predict([raw_value])[0])

        # Min–max normalization
        elif calibrator_spec.get("type") == "minmax":
            lower_bound = float(calibrator_spec["x_lo"])
            upper_bound = float(calibrator_spec["x_hi"])

            if upper_bound <= lower_bound:
                # Invalid calibrator — fall back to neutral midpoint
                calibrated_probabilities[category_name] = 50.0
            else:
                normalized_value = (raw_value - lower_bound) / (upper_bound - lower_bound)
                calibrated_probabilities[category_name] = float(np.clip(100.0 * normalized_value, 0.0, 100.0))

        # Fallback heuristic: crude linear scaling
        else:
            calibrated_probabilities[category_name] = float(np.clip(50.0 + 25.0 * raw_value, 0.0, 100.0))

    return calibrated_probabilities


def _iter_image_paths(images_dir: Path) -> list[Path]:
    """
    Gather all valid image paths from a directory.

    Recognizes .jpg, .jpeg, .png in any capitalization.

    :param images_dir: Directory containing images.
    :return: Sorted list of path objects.

    Example::
        In: _iter_image_paths(images_dir)
        Out: function result returned for provided inputs
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    log.debug("Scanning for images in: %s", images_dir)

    return sorted([image_path for image_path in Path(images_dir).glob("*") if image_path.suffix in valid_extensions])
