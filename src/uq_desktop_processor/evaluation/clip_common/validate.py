"""
Validates CLIP evaluation configuration and prepares per-category direction tensors.
"""

import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

import torch

try:
    import clip
except ImportError:
    clip = None
    _HAS_CLIP = False
else:
    _HAS_CLIP = True

log = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception storing validation errors and warnings.

    Example::
        In: raise ValidationError(["missing token"], ["beta is high"])
        Out: exception with .errors and .warnings attributes populated
    """

    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        """
        Initialize validation error with collected errors and optional warnings.

        :param errors: Validation error messages.
        :param warnings: Validation warnings gathered during checks.

        Example::
            In: ValidationError(["missing token"], ["beta is high"])
            Out: exception with .errors and .warnings attributes populated
        """
        super().__init__("\n".join(errors))
        self.errors = errors
        self.warnings = warnings or []


# Expected keys for prompt categories
SIDES: tuple[str, str] = ("pos", "neg")


def is_nonempty_string(value: Any) -> bool:
    """
    Check whether a value is a non-empty string.

    :param value: Any value to test.
    :return: True if value is a non-empty string, else False.

    Example::
        In: is_nonempty_string(value)
        Out: function result returned for provided inputs
    """
    return isinstance(value, str) and value.strip() != ""


def all_nonempty_strings(values: Iterable[Any]) -> bool:
    """
    Check whether all values in an iterable are non-empty strings.

    :param values: Iterable of values.
    :return: True if all are non-empty strings.

    Example::
        In: all_nonempty_strings(values)
        Out: function result returned for provided inputs
    """
    return all(is_nonempty_string(value) for value in values)


def as_string_set(values: Iterable[str]) -> set[str]:
    """
    Convert an iterable of strings into a cleaned set (trimmed, non-empty).

    :param values: Iterable of strings.
    :return: A set of stripped, valid strings.

    Example::
        In: as_string_set(values)
        Out: function result returned for provided inputs
    """
    return {value.strip() for value in values if is_nonempty_string(value)}


def list_or_empty(values: Iterable[Any] | None) -> list[Any]:
    """
    Convert an iterable to a list, or return an empty list if None.

    :param values: Iterable or None.
    :return: List of items or empty list.

    Example::
        In: list_or_empty(values)
        Out: function result returned for provided inputs
    """
    return list(values) if values is not None else []


def is_percentage(value: Any) -> bool:
    """
    Check whether a value is a numeric percentage between 0 and 100.

    :param value: Any value.
    :return: True if within a valid percentage range.

    Example::
        In: is_percentage(value)
        Out: function result returned for provided inputs
    """
    return isinstance(value, int | float) and 0.0 <= float(value) <= 100.0


def check_image_folder(image_folder_path: str) -> tuple[list[str], list[str]]:
    """
    Validate a folder containing images.

    :param image_folder_path: Path to folder.
    :return: (errors, warnings)

    Example::
        In: check_image_folder(image_folder_path)
        Out: function result returned for provided inputs
    """
    errors_list: list[str] = []
    warnings_list: list[str] = []

    log.debug("Validating image folder path: %s", image_folder_path)

    # Validate path input
    if not is_nonempty_string(image_folder_path):
        error_message = "image_folder: empty path."
        log.error(error_message)
        errors_list.append(error_message)
        return errors_list, warnings_list

    # Check folder existence
    if not os.path.isdir(image_folder_path):
        error_message = f"image_folder: folder does not exist: {image_folder_path!r}."
        log.error(error_message)
        errors_list.append(error_message)
        return errors_list, warnings_list

    # Check folder readability and image files
    try:
        file_names = [
            file_name
            for file_name in os.listdir(image_folder_path)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        log.debug("Found %s valid image files in folder.", len(file_names))
    except PermissionError:
        error_message = f"image_folder: no read permission: {image_folder_path!r}."
        log.error(error_message)
        errors_list.append(error_message)
        return errors_list, warnings_list

    if not file_names:
        error_message = f"image_folder: no .png/.jpg/.jpeg files found in {image_folder_path!r}."
        log.error(error_message)
        errors_list.append(error_message)

    return errors_list, warnings_list


def check_device(device_name: str) -> tuple[list[str], list[str]]:
    """
    Validate a PyTorch device string (CPU or CUDA).

    :param device_name: Device identifier string.
    :return: (errors, warnings)

    Example::
        In: check_device(device_name)
        Out: function result returned for provided inputs
    """
    errors_list: list[str] = []
    warnings_list: list[str] = []
    log.debug("Validating device configuration: %s", device_name)

    if not is_nonempty_string(device_name):
        errors_list.append("device: empty string.")
        return errors_list, warnings_list

    # CPU always valid
    if device_name == "cpu":
        return errors_list, warnings_list

    # CUDA validation
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            error_message = "device: 'cuda' specified but CUDA is not available."
            log.error(error_message)
            errors_list.append(error_message)
            return errors_list, warnings_list

        # Check GPU index if provided (e.g., 'cuda:1')
        if ":" in device_name:
            _, _, index_str = device_name.partition(":")
            try:
                index_int = int(index_str)
                gpu_count = torch.cuda.device_count()
                if not (0 <= index_int < gpu_count):
                    error_message = f"device: GPU index out of range (0..{gpu_count - 1})."
                    log.error(error_message)
                    errors_list.append(error_message)
            except ValueError:
                errors_list.append("device: invalid format after 'cuda:'.")
        return errors_list, warnings_list

    # Unknown device > warning only
    warning_message = f"device: unknown identifier '{device_name}', PyTorch will attempt to use it automatically."
    log.warning(warning_message)
    warnings_list.append(warning_message)
    return errors_list, warnings_list


def check_model_names(model_name_iterable: Iterable[str]) -> tuple[list[str], list[str]]:
    """
    Validate model name list for CLIP models.

    :param model_name_iterable: Iterable of model names.
    :return: (errors, warnings)

    Example::
        In: check_model_names(model_name_iterable)
        Out: function result returned for provided inputs
    """
    errors_list: list[str] = []
    warnings_list: list[str] = []

    names_list = list(model_name_iterable) if model_name_iterable is not None else []
    log.debug("Validating model names: %s", names_list)

    if not names_list:
        errors_list.append("model_names: empty list or tuple.")
        return errors_list, warnings_list

    if not all_nonempty_strings(names_list):
        errors_list.append("model_names: all elements must be non-empty strings.")

    # Check availability if CLIP is installed
    if _HAS_CLIP:
        try:
            available_models_set = set(clip.available_models())
            unknown_names = [name for name in names_list if name not in available_models_set]
            if unknown_names:
                warning_message = f"model_names: unsupported by clip: {', '.join(unknown_names)}."
                log.warning(warning_message)
                warnings_list.append(warning_message)
        except (OSError, RuntimeError, AttributeError):
            log.warning("Failed to retrieve available CLIP models list.")
            warnings_list.append("model_names: failed to read clip.available_models().")
    else:
        log.debug("CLIP module not loaded, skipping model name availability check.")
        warnings_list.append("model_names: 'clip' module not loaded — skipping availability check.")

    return errors_list, warnings_list


def check_beta_sigmoid(beta_value: float) -> tuple[list[str], list[str]]:
    """
    Validate the beta parameter used in the sigmoid probability function.

    :param beta_value: Numeric beta parameter.
    :return: (errors, warnings)

    Example::
        In: check_beta_sigmoid(beta_value)
        Out: function result returned for provided inputs
    """
    errors_list: list[str] = []
    warnings_list: list[str] = []

    if not isinstance(beta_value, int | float):
        errors_list.append("beta_sigmoid: must be a number.")
        return errors_list, warnings_list

    if beta_value <= 0:
        errors_list.append("beta_sigmoid: must be > 0.")
    elif beta_value > 100:
        warning_message = f"beta_sigmoid: {beta_value} > 100 — the curve will be extremely steep."
        log.warning(warning_message)
        warnings_list.append(warning_message)

    return errors_list, warnings_list


def check_order(category_order: Iterable[str]) -> tuple[list[str], list[str]]:
    """
    Validate the order of categories used in scoring.

    :param category_order: Iterable of category names.
    :return: (errors, warnings)

    Example::
        In: check_order(category_order)
        Out: function result returned for provided inputs
    """
    errors_list: list[str] = []
    warnings_list: list[str] = []
    order_list = list_or_empty(category_order)

    if not order_list:
        errors_list.append("order: cannot be empty.")
        return errors_list, warnings_list

    if not all_nonempty_strings(order_list):
        errors_list.append("order: all category names must be non-empty strings.")

    if len(set(order_list)) != len(order_list):
        errors_list.append("order: duplicate categories are not allowed.")

    return errors_list, warnings_list


def check_weights(weight_mapping: Mapping[str, float]) -> tuple[list[str], list[str]]:
    """
    Validate category weight configuration.

    :param weight_mapping: Mapping category → weight value.
    :return: (errors, warnings)

    Example::
        In: check_weights(weight_mapping)
        Out: function result returned for provided inputs
    """
    errors_list: list[str] = []
    warnings_list: list[str] = []

    if not isinstance(weight_mapping, Mapping):
        errors_list.append("weights: must be Mapping[str, float].")
        return errors_list, warnings_list

    if not weight_mapping:
        errors_list.append("weights: cannot be empty.")
        return errors_list, warnings_list

    # Validate each weight
    for category_key, weight_value in weight_mapping.items():
        if not is_nonempty_string(category_key):
            errors_list.append("weights: category key cannot be empty.")
        if not isinstance(weight_value, int | float):
            errors_list.append(f"weights[{category_key!r}]: must be a number.")
        elif weight_value <= 0:
            errors_list.append(f"weights[{category_key!r}]: must be > 0 (geometric mean).")
        elif not (float(weight_value) < float("inf")):
            errors_list.append(f"weights[{category_key!r}]: Inf/NaN values are not allowed.")

    # Aggregate checks
    try:
        total_weight = sum(float(value) for value in weight_mapping.values())
        if total_weight <= 0:
            errors_list.append("weights: the sum of weights must be > 0.")

        # Warn if one weight dominates the rest
        if weight_mapping and max(weight_mapping.values()) >= 10 * (total_weight / len(weight_mapping)):
            warning_message = "weights: one weight dominates (>= 10× the average)."
            log.warning(warning_message)
            warnings_list.append(warning_message)
    except (ValueError, TypeError):
        pass

    return errors_list, warnings_list


def check_prompts(prompt_mapping: Mapping[str, dict[str, Iterable[str]]]) -> tuple[list[str], list[str]]:
    """
    Validate the prompt structure (categories with 'pos'/'neg' lists).

    :param prompt_mapping: Mapping category → {'pos': [...], 'neg': [...]}
    :return: (errors, warnings)

    Example::
        In: check_prompts(prompt_mapping)
        Out: function result returned for provided inputs
    """
    errors_list: list[str] = []
    warnings_list: list[str] = []

    if not isinstance(prompt_mapping, Mapping) or not prompt_mapping:
        errors_list.append("prompts: must be a non-empty Mapping.")
        return errors_list, warnings_list

    for category_name, side_mapping in prompt_mapping.items():
        if not isinstance(side_mapping, Mapping):
            errors_list.append(f"prompts[{category_name!r}]: value must be a dict.")
            continue

        # Validate 'pos' and 'neg'
        for side_name in SIDES:
            if side_name not in side_mapping:
                errors_list.append(f"prompts[{category_name!r}]: missing key '{side_name}'.")
                continue

            values_list = list_or_empty(side_mapping[side_name])

            if len(values_list) == 0:
                errors_list.append(f"prompts[{category_name!r}]['{side_name}']: list cannot be empty.")
            elif not all_nonempty_strings(values_list):
                errors_list.append(
                    f"prompts[{category_name!r}]['{side_name}']: all elements must be non-empty strings."
                )

            # Warn about extremely long prompts (CLIP has token limits)
            very_long_prompts = [prompt for prompt in values_list if isinstance(prompt, str) and len(prompt) > 512]
            if very_long_prompts:
                warning_message = f"prompts[{category_name!r}]['{side_name}']: {len(very_long_prompts)} very long prompts (>512 characters)."
                log.warning(warning_message)
                warnings_list.append(warning_message)

    return errors_list, warnings_list


def check_keyset_consistency(
    category_order: Iterable[str],
    weight_mapping: Mapping[str, float],
    prompt_mapping: Mapping[str, dict[str, Iterable[str]]],
) -> tuple[list[str], list[str]]:
    """
    Ensure that the category names across 'order', 'weights', and 'prompts' match.

    :param category_order: Iterable of category names.
    :param weight_mapping: Mapping category → weight.
    :param prompt_mapping: Mapping category → prompt pairs.
    :return: (errors, warnings)

    Example::
        In: check_keyset_consistency(category_order, weight_mapping, prompt_mapping)
        Out: function result returned for provided inputs
    """
    errors_list: list[str] = []
    warnings_list: list[str] = []

    order_set = as_string_set(category_order)
    weights_set = as_string_set(weight_mapping.keys())
    prompts_set = as_string_set(prompt_mapping.keys())

    # All three sets must match exactly
    if not (order_set == weights_set == prompts_set):
        error_message = f"inconsistent keys: order={sorted(order_set)}, weights={sorted(weights_set)}, prompts={sorted(prompts_set)} — sets should match."
        log.error(error_message)
        errors_list.append(error_message)

    return errors_list, warnings_list


def validate_config(
    *,
    image_folder: str,
    device: str,
    model_names: Iterable[str],
    beta_sigmoid: float,
    order: Iterable[str],
    weights: Mapping[str, float],
    prompts: Mapping[str, dict[str, Iterable[str]]],
    raise_on_error: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Validate all configuration components and optionally raise an error.

    :param image_folder: Path to folder with images.
    :param device: PyTorch device string.
    :param model_names: List of CLIP model names.
    :param beta_sigmoid: Sigmoid scaling factor.
    :param order: Category order.
    :param weights: Category weights.
    :param prompts: Category → prompt structure.
    :param raise_on_error: Raise exception if errors found.
    :return: (errors, warnings)

    Example::
        In: validate_config()
        Out: function result returned for provided inputs
    """
    log.info("Starting configuration validation...")
    errors_accumulated: list[str] = []
    warnings_accumulated: list[str] = []

    # List of validator functions to run
    check_functions = [
        lambda: check_image_folder(image_folder),
        lambda: check_device(device),
        lambda: check_model_names(model_names),
        lambda: check_beta_sigmoid(beta_sigmoid),
        lambda: check_order(order),
        lambda: check_weights(weights),
        lambda: check_prompts(prompts),
        lambda: check_keyset_consistency(order, weights, prompts),
    ]

    # Aggregate errors and warnings from all checks
    for check_function in check_functions:
        check_errors, check_warnings = check_function()
        errors_accumulated.extend(check_errors)
        warnings_accumulated.extend(check_warnings)

    # Remove duplicates while preserving order
    errors_accumulated = list(dict.fromkeys(errors_accumulated))
    warnings_accumulated = list(dict.fromkeys(warnings_accumulated))

    if warnings_accumulated:
        log.info(
            "Configuration validation produced %s warning(s).",
            len(warnings_accumulated),
        )

    # Raise exception if configured to do so
    if errors_accumulated:
        log.error(
            "Configuration validation failed with %s error(s).",
            len(errors_accumulated),
        )
        if raise_on_error:
            raise ValidationError(errors_accumulated, warnings_accumulated)
    else:
        log.info("Configuration validation passed.")

    return errors_accumulated, warnings_accumulated
