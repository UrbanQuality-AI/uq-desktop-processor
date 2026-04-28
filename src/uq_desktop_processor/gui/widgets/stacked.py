"""
Generic stacked page container with a sidebar list and content widget swapper.
"""

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget


class CompactStackedWidget(QStackedWidget):
    """
    ``QStackedWidget`` that sizes vertically to the *current* page only.

    The default stacked widget reserves the height of the *tallest* page, which
    leaves a large empty gap when a shorter page (e.g. single line edit) is shown.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Run   init  .

        :param parent: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: __init__(parent)
            Out: UI/application state updated as intended.
        """
        super().__init__(parent)
        self.currentChanged.connect(self._on_page_changed)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def _on_page_changed(self, _index: int) -> None:
        """
        Run  on page changed.

        :param _index: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _on_page_changed(_index)
            Out: UI/application state updated as intended.
        """
        self.updateGeometry()
        parent_widget = self.parentWidget()
        if parent_widget is not None:
            parent_widget.updateGeometry()

    def sizeHint(self) -> QSize:  # noqa: N802  (Qt naming)
        """
        Run sizeHint.

        :return: Result of this step or updated UI/application state.

        Example::
            In: sizeHint()
            Out: UI/application state updated as intended.
        """
        w = self.currentWidget()
        if w is not None:
            cw = w.sizeHint()
            base_w = super().sizeHint().width()
            return QSize(max(base_w, cw.width()), cw.height())
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """
        Run minimumSizeHint.

        :return: Result of this step or updated UI/application state.

        Example::
            In: minimumSizeHint()
            Out: UI/application state updated as intended.
        """
        w = self.currentWidget()
        if w is not None:
            cm = w.minimumSizeHint()
            base_w = super().minimumSizeHint().width()
            return QSize(max(base_w, cm.width()), cm.height())
        return super().minimumSizeHint()


class NeonPanel(QFrame):
    """Titled panel with a vertical layout."""

    def __init__(self, title: str, subtitle: str | None = None) -> None:
        """
        Run   init  .

        :param title: See caller/context.
        :param subtitle: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: __init__(title, subtitle)
            Out: UI/application state updated as intended.
        """
        super().__init__()
        self.main_layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("PanelTitle")
        self.main_layout.addWidget(self.title_label)
        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setObjectName("PanelSubtitle")
            self.subtitle_label.setWordWrap(True)
            self.main_layout.addWidget(self.subtitle_label)
