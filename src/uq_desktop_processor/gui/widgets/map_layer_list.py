"""
QListWidget-based layer list with reordering and visibility for map stacks.
"""

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QListWidget, QWidget

log = logging.getLogger(__name__)


class MapLayerListWidget(QListWidget):
    """
    MapLayerListWidget UI helper class.
    """

    reordered = Signal()

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
        self._drag_src_row: int = -1

    def startDrag(self, supported_actions: Qt.DropAction) -> None:
        """
        Run startDrag.

        :param supported_actions: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: startDrag(supported_actions)
            Out: UI/application state updated as intended.
        """
        self._drag_src_row = self.currentRow()
        super().startDrag(supported_actions)

    def dropEvent(self, event: QDropEvent) -> None:
        """
        Run dropEvent.

        :param event: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: dropEvent(event)
            Out: UI/application state updated as intended.
        """
        try:
            src_row = self._drag_src_row if self._drag_src_row >= 0 else self.currentRow()
            if src_row < 0:
                super().dropEvent(event)
                return

            drop_position = event.position().toPoint()
            drop_item = self.itemAt(drop_position)

            if drop_item is None:
                dst_row = self.count()
            else:
                drop_item_rect = self.visualItemRect(drop_item)
                y_mid = drop_item_rect.y() + drop_item_rect.height() / 2.0
                dst_row = self.row(drop_item)
                if drop_position.y() >= y_mid:
                    dst_row += 1

            if dst_row > src_row:
                dst_row -= 1

            if dst_row == src_row:
                event.acceptProposedAction()
                return

            moved_item = self.takeItem(src_row)
            if moved_item is None:
                return

            self.insertItem(dst_row, moved_item)
            self.setCurrentRow(dst_row)
            self.reordered.emit()

            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()

        except (AttributeError, TypeError, RuntimeError) as error:
            log.error("Drop error: %s", error)
            super().dropEvent(event)
        finally:
            self._drag_src_row = -1
