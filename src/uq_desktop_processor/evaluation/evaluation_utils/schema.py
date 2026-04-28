"""
TypedDict schemas and helpers for structured CLIP (and related) evaluation JSON output.
"""

from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict


class CategoryScore(TypedDict):
    """
    CategoryScore evaluation helper class.

    Example::
        In: CategoryScore(...)
        Out: initialized instance ready for use
    """

    probability_pct: float
    delta: float | None


class ImageResult(TypedDict):
    """
    ImageResult evaluation helper class.

    Example::
        In: ImageResult(...)
        Out: initialized instance ready for use
    """

    filename: str
    image_path: NotRequired[str]
    overall_pct: float
    categories: Mapping[str, CategoryScore]


class EvaluationResult(TypedDict):
    """
    EvaluationResult evaluation helper class.

    Example::
        In: EvaluationResult(...)
        Out: initialized instance ready for use
    """

    model_name: str | None
    order: Sequence[str]
    weights: NotRequired[Mapping[str, float] | None]
    beta_sigmoid: float | None
    images: list[ImageResult]
    average_overall_pct: float | None
    skipped_images: NotRequired[list[str]]
    warnings: list[str]
    errors: list[str] | None
    # Optional extension point for future pipelines
    extra: NotRequired[Mapping[str, Any]]


def make_category_score(*, probability_pct: float, delta: float | None) -> CategoryScore:
    """
    Build one category score entry.

    :param probability_pct: Category probability in percent.
    :param delta: Raw directional score, if available.
    :return: Typed category score dictionary.

    Example::
        In: make_category_score(probability_pct=62.5, delta=0.12)
        Out: {"probability_pct": 62.5, "delta": 0.12}
    """
    return {"probability_pct": float(probability_pct), "delta": None if delta is None else float(delta)}


def make_image_result(
    *,
    filename: str,
    image_path: str | None = None,
    overall_pct: float,
    categories: Mapping[str, CategoryScore],
) -> ImageResult:
    """
    Build one image-level result record.

    :param filename: Source image path or name.
    :param image_path: Optional absolute/relative path to source image.
    :param overall_pct: Overall score in percent.
    :param categories: Per-category score mapping.
    :return: Typed image result dictionary.

    Example::
        In: make_image_result(filename="img.jpg", overall_pct=57.0, categories={"wealthier": {"probability_pct": 60.0, "delta": 0.2}})
        Out: {"filename": "img.jpg", "overall_pct": 57.0, "categories": {...}}
    """
    out: ImageResult = {"filename": str(filename), "overall_pct": float(overall_pct), "categories": categories}
    if image_path is not None:
        out["image_path"] = str(image_path)
    return out


def make_evaluation_result(
    *,
    model_name: str | None,
    order: Sequence[str],
    beta_sigmoid: float | None,
    images: list[ImageResult],
    average_overall_pct: float | None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    weights: Mapping[str, float] | None = None,
    skipped_images: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    """
    Assemble full evaluation payload in a consistent schema.

    :param model_name: Model identifier.
    :param order: Category output order.
    :param beta_sigmoid: Beta value used by sigmoid scoring (if applicable).
    :param images: Per-image results.
    :param average_overall_pct: Dataset average score.
    :param warnings: Non-fatal warnings.
    :param errors: Fatal or aggregated errors.
    :param weights: Optional category weights.
    :param skipped_images: Optional skipped-image messages.
    :param extra: Optional extension block.
    :return: Typed evaluation result dictionary.

    Example::
        In: make_evaluation_result(model_name="ViT-L/14", order=("wealthier",), beta_sigmoid=30.0, images=[], average_overall_pct=None)
        Out: {"model_name": "ViT-L/14", "order": ("wealthier",), "images": [], ...}
    """
    out: EvaluationResult = {
        "model_name": model_name,
        "order": tuple(order),
        "beta_sigmoid": beta_sigmoid,
        "images": images,
        "average_overall_pct": average_overall_pct,
        "warnings": list(warnings or []),
        "errors": errors,
    }
    if weights is not None:
        out["weights"] = dict(weights)
    if skipped_images is not None:
        out["skipped_images"] = list(skipped_images)
    if extra is not None:
        out["extra"] = dict(extra)
    return out
