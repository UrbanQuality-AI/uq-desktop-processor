"""
Registers explorer tool modules and exposes metadata for the sidebar.
"""

from PySide6.QtWidgets import QButtonGroup, QSplitter, QStackedWidget, QWidget


class ModulesMixin:
    """
    ModulesMixin UI helper class.
    """

    tool_container: QWidget
    button_group: QButtonGroup
    stacked_tools: QStackedWidget
    main_splitter: QSplitter
    current_active_module: int

    def toggle_module_panel(self, index: int) -> None:
        """
        Run toggle module panel.

        :param index: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: toggle_module_panel(index)
            Out: UI/application state updated as intended.
        """
        is_visible = self.tool_container.isVisible()

        if is_visible and self.current_active_module == index:
            self.tool_container.setVisible(False)
            self.button_group.setExclusive(False)
            self.button_group.button(index).setChecked(False)
            self.button_group.setExclusive(True)
            self.current_active_module = -1
        else:
            self.stacked_tools.setCurrentIndex(index)
            self.tool_container.setVisible(True)

            sizes = self.main_splitter.sizes()
            if sizes and sizes[0] < 50:
                self.main_splitter.setSizes([320, 800, 320])

            self.current_active_module = index
