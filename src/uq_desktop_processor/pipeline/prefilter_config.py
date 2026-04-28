"""
Loads YAML/JSON prefilter settings and merges them with pipeline defaults.
"""

from typing import Any

from uq_desktop_processor.evaluation.clip_prefilter import defaults as clip_prefilter_defaults


def prefilter_prompts_from_config(config: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """
    Parse multiline ``prefilter_*_prompts`` strings into positive/negative prompt tuples.

    Falls back to bundled defaults when any side is missing.

    :param config: Pipeline config dictionary with optional ``prefilter_pos_prompts`` / ``prefilter_neg_prompts``.
    :return: Dict with ``"pos"`` and ``"neg"`` tuple values.

    Example::
        In: prefilter_prompts_from_config({"prefilter_pos_prompts": "green\\npark", "prefilter_neg_prompts": ""})
        Out: {"pos": ("green", "park"), "neg": (...defaults...)}
    """

    def lines(text: Any) -> tuple[str, ...]:
        # Accept only non-empty strings; ignore blank lines.
        if not text or not isinstance(text, str):
            return ()
        return tuple(line.strip() for line in text.splitlines() if line.strip())

    positive_prompts = lines(config.get("prefilter_pos_prompts"))
    negative_prompts = lines(config.get("prefilter_neg_prompts"))
    if not positive_prompts:
        positive_prompts = tuple(clip_prefilter_defaults.FILTER_PROMPTS["pos"])
    if not negative_prompts:
        negative_prompts = tuple(clip_prefilter_defaults.FILTER_PROMPTS["neg"])
    return {"pos": positive_prompts, "neg": negative_prompts}
