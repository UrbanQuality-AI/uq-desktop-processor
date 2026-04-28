"""
Validates prefilter configuration dicts before loading models or scanning images.
"""

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from uq_desktop_processor.evaluation.clip_common import (
    ValidationError,
    check_beta_sigmoid,
    check_device,
    check_image_folder,
    check_model_names,
)

log = logging.getLogger(__name__)


def _is_percentage(value: Any) -> bool:
    """
    Check whether a given value represents a valid percentage (0–100).

    :param value: Value to test.
    :return: True if value is numeric and in range [0, 100].

    Example::
        In: _is_percentage(value)
        Out: function result returned for provided inputs
    """
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return False
    return 0.0 <= numeric_value <= 100.0


def _check_filter_threshold(threshold_value: float) -> tuple[list[str], list[str]]:
    """
    Validate the numeric threshold used for filtering.
    Ensures the value is a number and within [0, 100].

    :param threshold_value: Threshold percentage.
    :return: (errors, warnings)

    Example::
        In: _check_filter_threshold(threshold_value)
        Out: function result returned for provided inputs
    """
    error_messages: list[str] = []
    warning_messages: list[str] = []

    if not isinstance(threshold_value, int | float):
        error_message = "filter_threshold: must be a number."
        log.error(error_message)
        error_messages.append(error_message)
        return error_messages, warning_messages

    if not _is_percentage(threshold_value):
        error_message = "filter_threshold: allowed range is [0, 100]."
        log.error(error_message)
        error_messages.append(error_message)

    return error_messages, warning_messages


def _check_filter_prompts(filter_prompt_mapping: Mapping[str, Iterable[str]]) -> tuple[list[str], list[str]]:
    """
    Validate the structure and content of filter prompts.
    Ensures presence of 'pos' and 'neg' keys and that all prompts are non-empty strings.

    :param filter_prompt_mapping: Mapping with keys 'pos' and 'neg' pointing to lists of prompts.
    :return: (errors, warnings)

    Example::
        In: _check_filter_prompts(filter_prompt_mapping)
        Out: function result returned for provided inputs
    """
    error_messages: list[str] = []
    warning_messages: list[str] = []

    if not isinstance(filter_prompt_mapping, Mapping):
        error_message = "filter_prompts: must be a Mapping."
        log.error(error_message)
        error_messages.append(error_message)
        return error_messages, warning_messages

    for prompt_side_name in ("pos", "neg"):
        if prompt_side_name not in filter_prompt_mapping:
            error_message = f"filter_prompts: missing key '{prompt_side_name}'."
            log.error(error_message)
            error_messages.append(error_message)
            continue

        prompt_values = list(filter_prompt_mapping[prompt_side_name] or [])
        if not prompt_values:
            error_message = f"filter_prompts['{prompt_side_name}']: list cannot be empty."
            log.error(error_message)
            error_messages.append(error_message)
        elif not all(isinstance(prompt_text, str) and prompt_text.strip() for prompt_text in prompt_values):
            error_message = f"filter_prompts['{prompt_side_name}']: all elements must be non-empty strings."
            log.error(error_message)
            error_messages.append(error_message)

    return error_messages, warning_messages


def _validate_prefilter_config(
    *,
    image_folder: str,
    device: str,
    model_names: Iterable[str],
    beta_sigmoid: float,
    filter_threshold: float,
    filter_prompts: Mapping[str, Iterable[str]],
    raise_on_error: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Validate the configuration parameters for the pre-filtering process.

    This function aggregates various validation checks for image sources, compute devices,
    CLIP model specifications, and filtering parameters (thresholds and prompts).

    :param image_folder: Path to the directory containing images to be processed.
    :param device: The compute device identifier (e.g., 'cpu', 'cuda').
    :param model_names: An iterable of CLIP model names to be utilized.
    :param beta_sigmoid: The beta coefficient used in the sigmoid scaling function.
    :param filter_threshold: The cutoff percentage [0, 100] for the filter.
    :param filter_prompts: A mapping containing 'pos' (positive) and 'neg' (negative) prompt lists.
    :param raise_on_error: If True, raises a ValidationError when errors occur. Defaults to True.
    :return: A tuple containing a list of error messages and a list of warning messages.
    :raises ValidationError: If errors are found and raise_on_error is True.

    Example::
        In: _validate_prefilter_config()
        Out: function result returned for provided inputs
    """
    log.debug("Validating pre-filter configuration...")
    error_messages: list[str] = []
    warning_messages: list[str] = []

    validation_checks = [
        lambda: check_image_folder(image_folder),
        lambda: check_device(device),
        lambda: check_model_names(model_names),
        lambda: check_beta_sigmoid(beta_sigmoid),
        lambda: _check_filter_threshold(filter_threshold),
        lambda: _check_filter_prompts(filter_prompts),
    ]

    for validation_function in validation_checks:
        function_errors, function_warnings = validation_function()
        error_messages += function_errors
        warning_messages += function_warnings

    # Remove duplicates while preserving order
    error_messages = list(dict.fromkeys(error_messages))
    warning_messages = list(dict.fromkeys(warning_messages))

    if error_messages:
        log.error("Pre-filter validation failed with %s errors.", len(error_messages))
        if raise_on_error:
            raise ValidationError(error_messages, warning_messages)
    else:
        log.debug("Pre-filter configuration valid.")

    return error_messages, warning_messages
