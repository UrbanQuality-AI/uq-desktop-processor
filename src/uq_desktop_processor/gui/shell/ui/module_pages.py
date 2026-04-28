"""
Builds or wires stacked module pages into the shell navigation model.
"""

from typing import TYPE_CHECKING

from uq_desktop_processor.gui.shell.ui.pages import clip_prefilter, drive_route, mapillary, sampling_points, vit_scoring
from uq_desktop_processor.gui.shell.ui.placeholders import add_placeholder_tool_page

if TYPE_CHECKING:
    from uq_desktop_processor.gui.shell.explorer import UrbanQualityAIExplorer


def setup_module_pages(explorer: "UrbanQualityAIExplorer") -> None:
    """
    Run setup module pages.

    :param explorer: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: setup_module_pages(explorer)
        Out: UI/application state updated as intended.
    """
    drive_route.add_drive_route_page(explorer)
    sampling_points.add_sampling_points_page(explorer)
    mapillary.add_mapillary_page(explorer)
    add_placeholder_tool_page(
        explorer.stacked_tools,
        explorer.run_buttons,
        title="Panoramas → views",
        description="This module is a placeholder and will be implemented later.",
    )
    clip_prefilter.add_clip_prefilter_page(explorer)
    vit_scoring.add_vit_scoring_page(explorer)
    add_placeholder_tool_page(
        explorer.stacked_tools,
        explorer.run_buttons,
        title="Export results",
        description="This module is a placeholder and will be implemented later.",
    )
    add_placeholder_tool_page(
        explorer.stacked_tools,
        explorer.run_buttons,
        title="Validation",
        description="This module is a placeholder and will be implemented later.",
    )
