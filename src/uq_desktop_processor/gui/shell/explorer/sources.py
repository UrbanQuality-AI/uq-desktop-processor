"""
Defines map layer sources and GeoJSON loading for the explorer map.
"""

from PySide6.QtWidgets import QComboBox, QLabel

from uq_desktop_processor.gui.shell.constants import SOURCE_INPUT_STACK_PAGE_ORDER
from uq_desktop_processor.gui.widgets import CompactStackedWidget


class SourceInputsMixin:
    """
    SourceInputsMixin UI helper class.
    """

    @staticmethod
    def _apply_source_input_page(
        combo: QComboBox,
        stack: CompactStackedWidget,
        field_label: QLabel,
        labels: dict[str, str],
    ) -> None:
        """
        Run  apply source input page.

        :param combo: See caller/context.
        :param stack: See caller/context.
        :param field_label: See caller/context.
        :param labels: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _apply_source_input_page(combo, stack, field_label, labels)
            Out: UI/application state updated as intended.
        """
        selected_source = combo.currentData()
        source_key = str(selected_source) if selected_source is not None else "place"
        try:
            page_index = SOURCE_INPUT_STACK_PAGE_ORDER.index(source_key)
        except ValueError:
            page_index = 0
            source_key = "place"
        stack.setCurrentIndex(page_index)
        field_label.setText(labels[source_key])
