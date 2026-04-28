"""
Modal dialogs for the explorer: paths, confirmations, and configuration prompts.
"""

from PySide6.QtWidgets import QFileDialog, QLineEdit, QWidget


class DialogsMixin:
    """
    DialogsMixin UI helper class.
    """

    def pick_file(self, target: QLineEdit, filt: str) -> None:
        """
        Run pick file.

        :param target: See caller/context.
        :param filt: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: pick_file(target, filt)
            Out: UI/application state updated as intended.
        """
        parent_widget: QWidget | None = self if isinstance(self, QWidget) else None
        path, _ = QFileDialog.getOpenFileName(parent_widget, "Select file", "", filt)
        if path:
            target.setText(path)

    def pick_folder(self, target: QLineEdit) -> None:
        """
        Run pick folder.

        :param target: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: pick_folder(target)
            Out: UI/application state updated as intended.
        """
        parent_widget: QWidget | None = self if isinstance(self, QWidget) else None
        path = QFileDialog.getExistingDirectory(parent_widget, "Select folder")
        if path:
            target.setText(path)

    def pick_save_points_layer(self, target: QLineEdit) -> None:
        """
        Run pick save points layer.

        :param target: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: pick_save_points_layer(target)
            Out: UI/application state updated as intended.
        """
        parent_widget: QWidget | None = self if isinstance(self, QWidget) else None
        path, _ = QFileDialog.getSaveFileName(
            parent_widget,
            "Save point layer",
            target.text() or "sampling_points.geojson",
            "GeoJSON (*.geojson);;GeoPackage (*.gpkg)",
        )
        if path:
            target.setText(path)

    def pick_save_gpkg(self, target: QLineEdit) -> None:
        """
        Run pick save gpkg.

        :param target: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: pick_save_gpkg(target)
            Out: UI/application state updated as intended.
        """
        parent_widget: QWidget | None = self if isinstance(self, QWidget) else None
        path, _ = QFileDialog.getSaveFileName(
            parent_widget,
            "Save GeoPackage layer",
            target.text() or "vit_finetuned_scores.gpkg",
            "GeoPackage (*.gpkg)",
        )
        if path:
            target.setText(path)
