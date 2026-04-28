"""
Shell-wide UI constants: titles, spacing keys, and shared resource identifiers.
"""

CLIP_MODEL_CHOICES: list[tuple[str, tuple[str, ...]]] = [
    ("ViT-L/14 @336px", ("ViT-L/14@336px",)),
    ("ViT-L/14", ("ViT-L/14",)),
    ("ViT-B/32", ("ViT-B/32",)),
    ("ViT-B/16", ("ViT-B/16",)),
]

# Combo userData order must match stacked pages in route/planner input stacks.
SOURCE_INPUT_STACK_PAGE_ORDER: tuple[str, ...] = ("place", "region", "roads")

EULER_SOURCE_FIELD_LABELS: dict[str, str] = {
    "place": "City / place",
    "region": "Region file",
    "roads": "Roads file",
}
PLANNER_SOURCE_FIELD_LABELS: dict[str, str] = {
    "place": "Target city / place",
    "region": "Region file",
    "roads": "Roads file",
}
