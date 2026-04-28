"""
Side panel UI for toggling and ordering PyDeck map layers.
"""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from uq_desktop_processor.gui.widgets.map_layer_list import MapLayerListWidget


class LayerPanel:
    """
    LayerPanel UI helper class.
    """

    _LAYER_ID_ROLE: int = int(Qt.ItemDataRole.UserRole)

    def __init__(
        self,
        parent: QWidget,
        *,
        layer_ids_top_first: list[str],
        layer_labels: dict[str, str],
        on_changed: Callable[[], None],
        on_reordered: Callable[[], None],
        on_move_up: Callable[[], None],
        on_move_down: Callable[[], None],
    ) -> None:
        """
        Run   init  .

        :param parent: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: __init__(parent)
            Out: UI/application state updated as intended.
        """
        self.layer_ids_top_first = layer_ids_top_first
        self._layer_labels = layer_labels
        self._on_changed = on_changed
        self._on_reordered = on_reordered

        panel = QFrame(parent)
        panel.setObjectName("MapLayerPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(6, 4, 6, 6)
        panel_layout.setSpacing(4)

        title = QLabel("Layers")
        title.setObjectName("MapLayerTitle")
        panel_layout.addWidget(title)

        self.list = MapLayerListWidget(panel)
        self.list.setObjectName("MapLayerList")
        self.list.setMaximumHeight(200)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.setDragDropOverwriteMode(False)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setSpacing(2)

        for lid in layer_ids_top_first:
            item = QListWidgetItem(layer_labels[lid])
            item.setData(self._LAYER_ID_ROLE, lid)
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(item)

        self.list.itemChanged.connect(lambda _it: self._on_changed())
        self.list.model().rowsMoved.connect(lambda *_a: self._on_reordered())
        self.list.reordered.connect(lambda *_a: self._on_reordered())
        panel_layout.addWidget(self.list, 1)

        order_row = QHBoxLayout()
        order_row.setSpacing(6)
        btn_up = QPushButton("▲ Up")
        btn_up.setObjectName("LayerOrderUp")
        btn_up.setToolTip("Higher in the list = drawn above lower layers")
        btn_up.clicked.connect(on_move_up)
        btn_down = QPushButton("▼ Down")
        btn_down.setObjectName("LayerOrderDown")
        btn_down.clicked.connect(on_move_down)
        order_row.addWidget(btn_up)
        order_row.addWidget(btn_down)
        panel_layout.addLayout(order_row)

        self.widget = panel

    def item_for_layer(self, lid: str) -> QListWidgetItem | None:
        """
        Run item for layer.

        :param lid: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: item_for_layer(lid)
            Out: UI/application state updated as intended.
        """
        for row in range(self.list.count()):
            it = self.list.item(row)
            if it is not None and it.data(self._LAYER_ID_ROLE) == lid:
                return it
        return None

    def layer_order_top_to_bottom(self) -> list[str]:
        """
        Run layer order top to bottom.

        :return: Result of this step or updated UI/application state.

        Example::
            In: layer_order_top_to_bottom()
            Out: UI/application state updated as intended.
        """
        out: list[str] = []
        for row in range(self.list.count()):
            it = self.list.item(row)
            if it is None:
                continue
            lid = it.data(self._LAYER_ID_ROLE)
            if isinstance(lid, str):
                out.append(lid)
        return out

    def layer_checked(self, lid: str) -> bool:
        """
        Run layer checked.

        :param lid: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: layer_checked(lid)
            Out: UI/application state updated as intended.
        """
        item = self.item_for_layer(lid)
        if item is None:
            return False
        return item.checkState() == Qt.CheckState.Checked

    @staticmethod
    def active_item_flags() -> Any:
        """
        Run active item flags.

        :return: Result of this step or updated UI/application state.

        Example::
            In: active_item_flags()
            Out: UI/application state updated as intended.
        """
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
        )

    @staticmethod
    def inactive_item_flags() -> Any:
        """
        Run inactive item flags.

        :return: Result of this step or updated UI/application state.

        Example::
            In: inactive_item_flags()
            Out: UI/application state updated as intended.
        """
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled
