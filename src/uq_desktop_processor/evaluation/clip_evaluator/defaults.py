"""
Default paths, model names, and hyperparameters for the CLIP image evaluator.
"""

from types import MappingProxyType

DEFAULT_IMAGE_FOLDER = "images/to_evaluate"
DEFAULT_DEVICE = "cuda"
DEFAULT_MODEL_NAMES = ("ViT-L/14@336px",)
DEFAULT_BETA_SIGMOID = 30.0

DEFAULT_ORDER = ("wealth", "safety", "beauty")
DEFAULT_WEIGHTS = MappingProxyType(
    {
        "wealth": 0.01,
        "safety": 1.5,
        "beauty": 0.01,
    }
)

DEFAULT_PROMPTS = MappingProxyType(
    {
        "wealth": {
            "pos": (
                "modern upscale neighborhood with well-maintained houses and clean sidewalks",
                "affluent residential street with manicured lawns and luxury cars parked neatly",
                "elegant suburban area with detached villas and landscaped gardens",
                "wealthy quiet street with new facades, large windows, and decorative lighting",
                "high-income residential zone with stone fences and premium architecture",
            ),
            "neg": (
                "run-down street with peeling paint and old damaged houses",
                "poor urban area with graffiti and broken sidewalks",
                "low-income neighborhood with abandoned or boarded-up buildings",
                "crowded residential block with cluttered yards and visible decay",
                "street showing signs of poverty, trash, and disrepair",
            ),
        },
        "safety": {
            "pos": (
                "peaceful residential street with clear visibility and no signs of danger",
                "calm, well-lit suburban neighborhood with families walking and children playing",
                "clean area with maintained houses and no graffiti or broken glass",
                "quiet street with orderly parked cars and good street lighting",
                "safe environment with tidy gardens, fences, and visible community care",
            ),
            "neg": (
                "dark alley with broken lights, graffiti, and litter on the ground",
                "street showing vandalism, smashed windows, or police tape",
                "abandoned neighborhood with derelict buildings and no people",
                "unsafe area with heavy shadows and visible neglect",
                "crime-ridden urban zone with damaged cars and boarded houses",
            ),
        },
        "beauty": {
            "pos": (
                "beautiful tree-lined residential street under warm sunlight",
                "charming neighborhood with colorful facades and blooming gardens",
                "scenic avenue with tidy sidewalks, greenery, and balanced composition",
                "aesthetically pleasing street with clean architecture and natural light",
                "picturesque suburban scene with flowers, cafés, and cozy atmosphere",
            ),
            "neg": (
                "dreary street under grey overcast sky with bare trees",
                "dirty industrial-looking road with trash and potholes",
                "unappealing neighborhood with dull colors and visual clutter",
                "ugly or neglected street lacking greenery and harmony",
                "bleak winter street with mud, slush, and poor lighting",
            ),
        },
    }
)
