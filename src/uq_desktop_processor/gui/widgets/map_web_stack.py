"""
Stacked Qt web views for switching between multiple embedded map instances.
"""

from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget


class MapWebStack(QWidget):
    """
    MapWebStack UI helper class.
    """

    _LAYER_CORNER_MARGIN = 10
    _LAYER_PANEL_MAX_WIDTH = 210

    def __init__(
        self,
        web: QWidget,
        overlay: QWidget,
        layer_panel: QWidget | None = None,
    ) -> None:
        """
        Run   init  .

        :param web: See caller/context.
        :param overlay: See caller/context.
        :param layer_panel: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: __init__(web, overlay, layer_panel)
            Out: UI/application state updated as intended.
        """
        super().__init__()
        self._web = web
        self._overlay = overlay
        self._layer_panel = layer_panel
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(web)
        overlay.setParent(self)
        if layer_panel is not None:
            layer_panel.setParent(self)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Run resizeEvent.

        :param event: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: resizeEvent(event)
            Out: UI/application state updated as intended.
        """
        super().resizeEvent(event)
        r = self.rect()
        self._overlay.setGeometry(r)
        if self._layer_panel is not None:
            lp = self._layer_panel
            m = self._LAYER_CORNER_MARGIN
            lp.adjustSize()
            hint = lp.sizeHint()
            w = max(lp.minimumWidth(), min(hint.width(), self._LAYER_PANEL_MAX_WIDTH))
            max_h = max(100, r.height() - 2 * m)
            h = min(hint.height(), max_h)
            x = r.x() + r.width() - w - m
            y = r.y() + m
            lp.setGeometry(x, y, w, h)
            lp.raise_()
        self._overlay.raise_()
