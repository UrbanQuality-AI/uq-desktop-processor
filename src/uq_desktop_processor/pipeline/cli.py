"""
CLI entry: runs the full UrbanQuality-AI pipeline (points, download, prefilter, evaluate, export).
"""

import logging

from uq_desktop_processor.pipeline.pipeline import UrbanQualityAIPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline() -> int:
    """
    Run the default end-to-end pipeline from the command line.

    :return: Process-like exit code (0 success, 1 operational error, 130 interrupted).

    Example::
        In: run_pipeline()
        Out: 0
    """
    pipeline = UrbanQualityAIPipeline()
    try:
        pipeline.step_1_generate_points()
        pipeline.step_2_download_images()

        prefilter_stats = pipeline.step_3_prefilter()

        # Skip scoring/export when prefilter removed all images.
        if prefilter_stats.get("summary", {}).get("kept", 0) > 0:
            pipeline.step_4_evaluate()
            pipeline.step_5_export()
        return 0
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError) as error:
        logger.error("Pipeline failed due to operational error: %s", error)
        return 1
