"""
Per-image CLIP scoring: sigmoid probabilities, categories, and optional weighting.
"""

import logging
import math
from collections.abc import Callable
from typing import Any, cast

import torch
from PIL import Image
from torch import Tensor, nn

from uq_desktop_processor.evaluation.clip_common.model import sigmoid_probability

log = logging.getLogger(__name__)


def _load_and_encode_image(
    path: str, preprocess: Callable[[Image.Image], Tensor], model: nn.Module, device: torch.device
) -> Tensor:
    """
    Load an image from disk, preprocess it, and return its normalized
    CLIP feature embedding.

    :param path: Path to the image file.
    :param preprocess: Preprocessing function associated with the CLIP model.
    :param model: CLIP model used for encoding images.
    :param device: Torch device on which computation is performed.
    :return: L2-normalized image embedding tensor (1D).

    Example::
        In: _load_and_encode_image(path, preprocess, model, device)
        Out: function result returned for provided inputs
    """
    log.debug("Loading and encoding image from: %s", path)

    try:
        # Load the image and convert to RGB
        image = Image.open(path).convert("RGB")

        # Apply CLIP preprocessing and move to device
        tensor = preprocess(image).unsqueeze(0).to(device)

        # Extract feature embedding
        model_any = cast(Any, model)
        features = model_any.encode_image(tensor)

        # Normalize the feature vector
        features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze()

    except Exception as error:
        log.error("Failed to process image '%s': %s", path, error)
        raise


def _compute_weighted_score(
    results: dict[str, dict[str, float]], categories: list[str], weights: dict[str, float]
) -> float:
    """
    Compute the overall score using a weighted geometric mean of probabilities.

    :param results: Dict of per-category scores, each containing:
                    { "delta": float, "probability": float }
    :param categories: List of category names in scoring order.
    :param weights: Mapping category → weight.
    :return: Weighted overall score as a percentage.

    Example::
        In: _compute_weighted_score(results, categories, weights)
        Out: function result returned for provided inputs
    """
    min_prob: float = 1e-6  # Avoid log(0) by clamping very small values
    weighted_log_sum: float = 0.0
    total_weight: float = 0.0

    for category in categories:
        # Probability is stored as percentage; convert to [0, 1] range
        prob_percent: float = results[category]["probability"]
        prob: float = max(prob_percent / 100.0, min_prob)

        weight: float = weights[category]

        # Use weighted geometric mean = exp(weighted average of log-probabilities)
        weighted_log_sum += weight * math.log(prob)
        total_weight += weight

    # Convert weighted log-mean back to percentage probability space.
    score: float = math.exp(weighted_log_sum / total_weight) * 100.0
    return score


@torch.no_grad()
def score_image(
    path: str,
    model: nn.Module,
    preprocess: Callable[[Image.Image], Tensor],
    directions: dict[str, Tensor],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Score a single image using CLIP by comparing it with category direction vectors.

    :param path: Path to the image file.
    :param model: CLIP model for encoding.
    :param preprocess: CLIP-specific preprocessing function.
    :param directions: Precomputed direction vectors per category.
    :param config: Scoring configuration dictionary containing:
                   - "device": torch.device
                   - "beta_sigmoid": float
                   - "order": list/tuple of categories
                   - "weights": weight mapping
    :return: Structured scoring result:
             {
                "category": { "delta": float, "probability": float },
                ...
                "overall": float
             }

    Example::
        In: score_image(path, model, preprocess, directions, config)
        Out: function result returned for provided inputs
    """
    device: torch.device = config["device"]
    beta: float = config["beta_sigmoid"]

    try:
        # Load and encode the image into CLIP's embedding space
        features: Tensor = _load_and_encode_image(path, preprocess, model, device)

        results: dict[str, Any] = {}

        # Compute per-category delta and probability
        for category in config["order"]:
            # Dot product measures similarity to direction vector
            delta: float = torch.dot(features, directions[category]).item()
            # Convert similarity into a probability using a sigmoid curve
            probability: float = sigmoid_probability(delta, beta)
            results[category] = {"delta": delta, "probability": probability}

        # Compute final overall score via weighted geometric mean
        overall: float = _compute_weighted_score(results, config["order"], config["weights"])

        results["overall"] = overall

        if log.isEnabledFor(logging.DEBUG):
            log.debug("Scored '%s' -> Overall: %.2f", path, overall)

        return results

    except Exception as error:
        log.error("Error while scoring image '%s': %s", path, error)
        raise
