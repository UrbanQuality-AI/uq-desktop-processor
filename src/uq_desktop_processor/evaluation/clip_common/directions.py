"""
Builds CLIP embedding direction vectors from positive/negative prompt pairs per category.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch
from PIL import Image
from torch import Tensor, nn

from .model import encode_texts

log = logging.getLogger(__name__)


def _normalize(vector: Tensor) -> Tensor:
    """
    Normalize a vector to unit length (L2 norm).
    Returns the original vector if its norm is zero.

    Example::
        In: _normalize(vector)
        Out: function result returned for provided inputs
    """
    vector_norm = vector.norm(p=2)
    return vector / vector_norm if vector_norm > 0 else vector


def prepare_directions(
    prompts: Mapping[str, Mapping[str, Iterable[str]]],
    model: nn.Module,
    device: torch.device | str,
) -> dict[str, Tensor]:
    """
    Compute semantic direction vectors for each category based on
    positive and negative prompt examples.

    :param prompts: Dictionary containing categories with "pos" and "neg" prompt lists.
    :param model: Text encoding model.
    :param device: Torch device for computations.
    :return: Dictionary mapping category names to normalized direction vectors.

    Example::
        In: prepare_directions(prompts, model, device)
        Out: function result returned for provided inputs
    """
    directions: dict[str, Tensor] = {}

    for category, prompt_sides in prompts.items():
        log.debug("Encoding prompts for category: '%s'", category)

        try:
            # Encode positive prompts and compute their average embedding
            positive_prompt_embedding = encode_texts(prompt_sides["pos"], model, device).mean(dim=0)
            # Encode negative prompts and compute their average embedding
            negative_prompt_embedding = encode_texts(prompt_sides["neg"], model, device).mean(dim=0)

            # Compute difference and normalize to get a direction vector
            direction = _normalize(positive_prompt_embedding - negative_prompt_embedding)
            directions[category] = direction
        except Exception as error:
            log.error("Failed to compute direction for category '%s': %s", category, error)
            raise

    log.debug("All direction vectors computed successfully.")

    return directions


@torch.no_grad()
def encode_image(
    img: Image.Image,
    preprocess: Callable[[Image.Image], Tensor],
    model: nn.Module,
    device: torch.device | str,
) -> Tensor:
    """
    Encode an image into a normalized feature vector using the model.

    :param img: Input PIL image.
    :param preprocess: Preprocessing function for the model.
    :param model: Vision encoder model.
    :param device: Torch device for computations.
    :return: Normalized image embedding tensor.

    Example::
        In: encode_image(img, preprocess, model, device)
        Out: function result returned for provided inputs
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Encoding image tensor (size: %s)...", img.size)

    # Convert image to RGB, preprocess, and move to device
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    # Extract image features
    model_any = cast(Any, model)
    features = model_any.encode_image(tensor)
    # Normalize the feature vector
    return (features / features.norm(dim=-1, keepdim=True)).squeeze()
