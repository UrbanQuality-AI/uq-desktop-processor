"""
Evaluation module:   init  .
"""

from .directions import encode_image, prepare_directions
from .model import encode_texts, load_model, sigmoid_probability
from .printer import print_results
from .validate import (
    # Classes and main functions
    ValidationError,
    all_nonempty_strings,
    as_string_set,
    check_beta_sigmoid,
    check_device,
    check_image_folder,
    check_keyset_consistency,
    check_model_names,
    check_order,
    check_prompts,
    check_weights,
    # Low-level validation functions (for flexibility)
    is_nonempty_string,
    is_percentage,
    list_or_empty,
    validate_config,
)

__all__ = [
    "ValidationError",
    "all_nonempty_strings",
    "as_string_set",
    "check_beta_sigmoid",
    "check_device",
    "check_image_folder",
    "check_keyset_consistency",
    "check_model_names",
    "check_order",
    "check_prompts",
    "check_weights",
    "encode_image",
    "encode_texts",
    "is_nonempty_string",
    "is_percentage",
    "list_or_empty",
    "load_model",
    "prepare_directions",
    "print_results",
    "sigmoid_probability",
    "validate_config",
]
