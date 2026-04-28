"""
Default prompts, thresholds, and folder names for CLIP-based image prefiltering.
"""

from types import MappingProxyType

DEFAULT_IMAGE_FOLDER = "images/to_filter"
DEFAULT_DEVICE = "cuda"
DEFAULT_MODEL_NAMES = ("ViT-L/14@336px",)

DEFAULT_BETA_SIGMOID = 30.0
DEFAULT_FILTER_THRESHOLD = 40.0

DEFAULT_REJECTED_FOLDER = "rejected"

FILTER_PROMPTS = MappingProxyType(
    {
        "pos": (
            "sunny day",
            "clear sky",
            "blue sky",
            "bright daylight",
            "good weather",
            "sunlit street",
            "dry road surface",
            "high visibility",
        ),
        "neg": (
            "rain",
            "rainy weather",
            "heavy rain",
            "drizzle",
            "wet road surface",
            "puddles on the road",
            "overcast sky",
            "dark cloudy sky",
            "fog",
            "snow",
        ),
    }
)
