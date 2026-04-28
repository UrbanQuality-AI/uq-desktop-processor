"""
Loads OpenAI CLIP models and provides text/image encoding for evaluation pipelines.
"""

import logging
import math
from collections.abc import Callable, Iterable
from typing import Any, cast

import clip
import torch
from PIL import Image

log = logging.getLogger(__name__)


def load_model(
    model_names: Iterable[str],
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor], str]:
    """
    Try loading a CLIP model from a list of model names.
    Returns the first successfully loaded model along with its preprocess function
    and the model name that was used.

    :param model_names: Iterable of CLIP model names to try loading.
    :param device: Device to load the model onto.
    :return: (model, preprocess_function, loaded_model_name)

    Example::
        In: load_model(model_names, device)
        Out: function result returned for provided inputs
    """
    log.info("Attempting to load CLIP model on device: %s", device)

    last_error = None
    for name in model_names:
        try:
            # Attempt to load a model using the CLIP API
            model, preprocess = clip.load(name, device=device, jit=False)
            return model.eval(), preprocess, name
        except RuntimeError as runtime_error:
            # Store the last error in case all models fail
            last_error = runtime_error
    # Raise an error if none of the models could be loaded
    raise RuntimeError("No model could be loaded into memory.") from last_error


@torch.no_grad()
def encode_texts(
    texts: str | Iterable[str],
    model: torch.nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Tokenize and encode one or multiple text inputs using a CLIP model.
    Returns L2-normalized text embeddings.

    :param texts: A single string or a sequence of text strings.
    :param model: The CLIP model used for encoding.
    :param device: Device to run the encoding on.
    :return: Normalized text embedding tensor of shape (N, D).

    Example::
        In: encode_texts(texts, model, device)
        Out: function result returned for provided inputs
    """
    # Ensure texts is a materialized sequence.
    text_list = [texts] if isinstance(texts, str) else list(texts)

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Encoding %s text prompt(s)...", len(text_list))

    # Tokenize text and move to device
    tokens = clip.tokenize(text_list, truncate=True).to(device)

    # Encode text using CLIP
    model_any = cast(Any, model)
    features = model_any.encode_text(tokens)

    # Return normalized embeddings
    return features / features.norm(dim=-1, keepdim=True)


def sigmoid_probability(
    delta: float | torch.Tensor,
    beta: float | torch.Tensor,
) -> float | torch.Tensor:
    """
    Compute a sigmoid-based probability scaled to 0–100 range.
    Accepts both scalars and torch tensors.

    :param delta: Input value (difference or score).
    :param beta: Scaling factor controlling steepness.
    :return: Probability between 0 and 100.

    Example::
        In: sigmoid_probability(delta, beta)
        Out: function result returned for provided inputs
    """
    # Tensor-based implementation if either input is a tensor
    if isinstance(delta, torch.Tensor) or isinstance(beta, torch.Tensor):
        delta_t = delta if isinstance(delta, torch.Tensor) else torch.tensor(delta)
        beta_t = beta if isinstance(beta, torch.Tensor) else torch.tensor(beta)
        return 100.0 / (1.0 + torch.exp(-beta_t * delta_t))

    # Fallback to pure Python math for scalar values
    return 100.0 / (1.0 + math.exp(-beta * delta))
