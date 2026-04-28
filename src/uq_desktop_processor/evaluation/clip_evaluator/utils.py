"""
Filesystem and formatting helpers used by the CLIP evaluator CLI run.
"""

from typing import Any

from uq_desktop_processor.evaluation.clip_common import print_results as _print_common_results


def print_results(result: dict[str, Any]) -> None:
    """
    Wrapper around the shared print_results function.

    This exists to provide a local entry point for printing evaluation
    results, while delegating the actual formatting and output logic
    to the shared printer utility.

    :param result: Result dictionary produced by the evaluation pipeline.
    :return: Whatever the underlying printer function returns.

    Example::
        In: print_results(result)
        Out: function result returned for provided inputs
    """
    return _print_common_results(result)
