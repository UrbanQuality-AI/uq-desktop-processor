"""
Default checkpoint paths and inference settings for fine-tuned ViT scoring.
"""

from collections.abc import Sequence

DEFAULT_CATEGORY_ORDER: Sequence[str] = (
    "safer",
    "wealthier",
    "more beautiful",
    "livelier",
    "less depressing",
    "less boring",
)
