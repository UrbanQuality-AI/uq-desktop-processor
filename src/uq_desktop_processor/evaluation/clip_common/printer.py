"""
Prints CLIP evaluation summaries and validation output to the console (e.g. tabulated).
"""

import logging

from tabulate import tabulate  # type: ignore[import-untyped]

log = logging.getLogger(__name__)


def _handle_errors(result: dict) -> bool:
    """
    Check whether the result dictionary contains validation errors.
    If errors exist, print them along with any warnings.

    :param result: Result dictionary potentially containing "errors" and "warnings".
    :return: True if errors were found (and processing should stop), otherwise False.

    Example::
        In: _handle_errors(result)
        Out: function result returned for provided inputs
    """
    errors = result.get("errors")
    if not errors:
        return False

    # Log validation errors
    log.error("Configuration validation failed:")
    for error_message in errors:
        log.error(" - %s", error_message)

    # Log warnings if present
    warnings = result.get("warnings") or []
    if warnings:
        log.warning("Warnings:")
        for warning_message in warnings:
            log.warning(" - %s", warning_message)

    return True


def _build_rows(images: list, order: list[str]) -> list[list[str]]:
    """
    Build table rows for output based on image results and category order.

    :param images: List of image result dictionaries.
    :param order: List of category names in display order.
    :return: List of formatted table rows.

    Example::
        In: _build_rows(images, order)
        Out: function result returned for provided inputs
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Formatting results table for %s images...", len(images))

    rows = []
    for image_result in images:
        # Start each row with the filename
        row = [image_result["filename"]]

        # Add delta and probability for each category
        for category_name in order:
            category_data = image_result["categories"][category_name]
            delta = category_data.get("delta")
            delta_cell = f"{delta:+.3f}" if isinstance(delta, int | float) else "   n/a"
            row += [
                delta_cell,  # Signed delta value (or n/a for non-CLIP pipelines)
                f"{category_data['probability_pct']:6.1f}%",  # Probability percentage
            ]

        # Add overall score
        row.append(f"{image_result['overall_pct']:6.1f}%")
        rows.append(row)

    # Sort rows by overall score in descending order
    rows.sort(key=lambda row_values: float(row_values[-1].rstrip("%")), reverse=True)
    return rows


def _build_headers(order: list[str]) -> list[str]:
    """
    Build column headers for the results table based on category order.

    :param order: List of category names.
    :return: List of formatted column headers.

    Example::
        In: _build_headers(order)
        Out: function result returned for provided inputs
    """
    return (
        ["filename"]
        + [
            # Even index > value column, odd index > percentage column
            f"{category_name} value" if index % 2 == 0 else f"{category_name} %"
            for index, category_name in enumerate(order * 2)
        ]
        + ["overall %"]
    )


def _log_warnings(warnings: list[str] | None) -> None:
    """
    Log warnings if any exist.

    :param warnings: List of warning messages or None.

    Example::
        In: _log_warnings(warnings)
        Out: function result returned for provided inputs
    """
    if not warnings:
        return

    log.warning("[WARNINGS]")
    for warning_message in warnings:
        log.warning(" - %s", warning_message)


def print_results(result: dict) -> None:
    """
    Print formatted results using 'tabulate', including scores per image,
    model name, overall average, and warnings.

    :param result: Dictionary containing processed evaluation results.

    Example::
        In: print_results(result)
        Out: function result returned for provided inputs
    """
    # Stop if validation errors were detected
    if _handle_errors(result):
        return

    images = result.get("images", [])
    order = result.get("order", [])
    model_name = result.get("model_name", "unknown")

    if not images:
        print("No results to display.")
        return

    rows = _build_rows(images, order)
    headers = _build_headers(order)

    print(f"\nResults using model: {model_name}")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # Print average overall score if available
    average_overall_pct = result.get("average_overall_pct")
    if average_overall_pct is not None:
        print(f"\nAverage overall score: {average_overall_pct:.1f}%")

    # Print any warnings
    _log_warnings(result.get("warnings"))
