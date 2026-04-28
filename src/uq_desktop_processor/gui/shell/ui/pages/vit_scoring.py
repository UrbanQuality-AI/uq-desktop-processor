"""
Stacked tool page: fine-tuned ViT scoring configuration and batch runs.
"""

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from uq_desktop_processor.gui.widgets import NeonPanel

if TYPE_CHECKING:
    from uq_desktop_processor.gui.shell.explorer import UrbanQualityAIExplorer


def add_vit_scoring_page(explorer: "UrbanQualityAIExplorer") -> None:
    """
    Run add vit scoring page.

    :param explorer: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: add_vit_scoring_page(explorer)
        Out: UI/application state updated as intended.
    """
    p5 = QWidget()
    l5 = QVBoxLayout(p5)
    pan5 = NeonPanel(
        "ViT scoring (fine-tuned)",
        "Score images with a fine-tuned ViT model and write a GeoPackage layer.",
    )
    f5 = QFormLayout()

    imgs_row = QHBoxLayout()
    explorer.vit_images_dir_edit = QLineEdit("")
    explorer.vit_images_dir_edit.setPlaceholderText("folder with images (required)")
    btn_imgs = QPushButton("…")
    btn_imgs.setFixedWidth(36)
    btn_imgs.clicked.connect(lambda: explorer.pick_folder(explorer.vit_images_dir_edit))
    imgs_row.addWidget(explorer.vit_images_dir_edit)
    imgs_row.addWidget(btn_imgs)
    f5.addRow("Images folder", imgs_row)

    model_row = QHBoxLayout()
    explorer.vit_model_path_edit = QLineEdit("")
    explorer.vit_model_path_edit.setPlaceholderText("model weights file (.pt/.pth) (required)")
    btn_model = QPushButton("…")
    btn_model.setFixedWidth(36)
    btn_model.clicked.connect(lambda: explorer.pick_file(explorer.vit_model_path_edit, "Model (*.pt *.pth)"))
    model_row.addWidget(explorer.vit_model_path_edit)
    model_row.addWidget(btn_model)
    f5.addRow("Model path", model_row)

    cal_row = QHBoxLayout()
    explorer.vit_calibrators_edit = QLineEdit("")
    explorer.vit_calibrators_edit.setPlaceholderText("(optional) folder with calibrators; empty = calibration OFF")
    btn_cal = QPushButton("…")
    btn_cal.setFixedWidth(36)
    btn_cal.clicked.connect(lambda: explorer.pick_folder(explorer.vit_calibrators_edit))
    cal_row.addWidget(explorer.vit_calibrators_edit)
    cal_row.addWidget(btn_cal)
    f5.addRow("Calibrators dir", cal_row)

    vit_out_row = QHBoxLayout()
    explorer.vit_output_path_edit = QLineEdit("")
    explorer.vit_output_path_edit.setPlaceholderText("e.g. results/vit_scores.gpkg (include .gpkg)")
    btn_vit_out = QPushButton("…")
    btn_vit_out.setFixedWidth(36)
    btn_vit_out.clicked.connect(lambda: explorer.pick_save_gpkg(explorer.vit_output_path_edit))
    vit_out_row.addWidget(explorer.vit_output_path_edit)
    vit_out_row.addWidget(btn_vit_out)
    f5.addRow("Output layer (.gpkg)", vit_out_row)

    explorer.vit_output_layer_name_edit = QLineEdit("vit_finetuned_scores")
    f5.addRow("Layer name", explorer.vit_output_layer_name_edit)

    explorer.vit_model_name_edit = QLineEdit("vit_base_patch14_dinov2.lvd142m")
    f5.addRow("Model name", explorer.vit_model_name_edit)

    explorer.vit_image_size_spin = QSpinBox()
    explorer.vit_image_size_spin.setRange(128, 1024)
    explorer.vit_image_size_spin.setValue(224)
    f5.addRow("Image size", explorer.vit_image_size_spin)

    explorer.vit_batch_size_spin = QSpinBox()
    explorer.vit_batch_size_spin.setRange(1, 256)
    explorer.vit_batch_size_spin.setValue(32)
    f5.addRow("Batch size", explorer.vit_batch_size_spin)

    explorer.vit_device_combo = QComboBox()
    explorer.vit_device_combo.addItem("Auto (CUDA if FORCE_CUDA=1)", "auto")
    explorer.vit_device_combo.addItem("CPU", "cpu")
    explorer.vit_device_combo.addItem("CUDA", "cuda")
    f5.addRow("Torch device", explorer.vit_device_combo)

    pan5.main_layout.addLayout(f5)
    pan5.main_layout.addStretch()
    btn5 = QPushButton("EVALUATE (ViT) ⚙")
    btn5.setObjectName("RunBtn")
    btn5.clicked.connect(explorer.on_run_vit_evaluate)
    pan5.main_layout.addWidget(btn5)
    l5.addWidget(pan5)
    explorer.stacked_tools.addWidget(p5)
    explorer.run_buttons.append(btn5)
