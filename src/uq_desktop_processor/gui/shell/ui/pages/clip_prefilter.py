"""
Stacked tool page: CLIP prefilter (configure and run image filtering).
"""

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from uq_desktop_processor.evaluation.clip_prefilter.defaults import FILTER_PROMPTS as CLIP_PREFILTER_DEFAULT_PROMPTS
from uq_desktop_processor.gui.shell.constants import CLIP_MODEL_CHOICES
from uq_desktop_processor.gui.widgets import NeonPanel

if TYPE_CHECKING:
    from uq_desktop_processor.gui.shell.explorer import UrbanQualityAIExplorer


def add_clip_prefilter_page(explorer: "UrbanQualityAIExplorer") -> None:
    """
    Run add clip prefilter page.

    :param explorer: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: add_clip_prefilter_page(explorer)
        Out: UI/application state updated as intended.
    """
    prefilter_page = QWidget()
    page_layout = QVBoxLayout(prefilter_page)
    prefilter_panel = NeonPanel(
        "CLIP prefilter",
        "Compare images to two phrase lists and move low-score matches to the rejected folder.",
    )
    form_layout = QFormLayout()

    pre_in_row = QHBoxLayout()
    explorer.prefilter_image_folder_edit = QLineEdit(str(Path("data") / "images" / "raw"))
    explorer.prefilter_image_folder_edit.setPlaceholderText("Folder with .jpg / .png to filter")
    btn_pf_in = QPushButton("…")
    btn_pf_in.setFixedWidth(36)
    btn_pf_in.clicked.connect(lambda: explorer.pick_folder(explorer.prefilter_image_folder_edit))
    pre_in_row.addWidget(explorer.prefilter_image_folder_edit)
    pre_in_row.addWidget(btn_pf_in)
    form_layout.addRow("Input: images folder", pre_in_row)

    pre_rej_row = QHBoxLayout()
    explorer.prefilter_rejected_edit = QLineEdit("rejected")
    explorer.prefilter_rejected_edit.setPlaceholderText("Sibling name (e.g. rejected) or full folder path")
    btn_pf_rej = QPushButton("…")
    btn_pf_rej.setFixedWidth(36)
    btn_pf_rej.clicked.connect(lambda: explorer.pick_folder(explorer.prefilter_rejected_edit))
    pre_rej_row.addWidget(explorer.prefilter_rejected_edit)
    pre_rej_row.addWidget(btn_pf_rej)
    form_layout.addRow("Output: rejected folder", pre_rej_row)

    explorer.model_combo = QComboBox()
    for label, names in CLIP_MODEL_CHOICES:
        explorer.model_combo.addItem(label, names)
    form_layout.addRow("CLIP model", explorer.model_combo)

    explorer.device_combo = QComboBox()
    explorer.device_combo.addItem("Auto (CUDA if FORCE_CUDA=1)", "auto")
    explorer.device_combo.addItem("CPU", "cpu")
    explorer.device_combo.addItem("CUDA", "cuda")
    form_layout.addRow("Torch device", explorer.device_combo)

    explorer.threshold_slider = QSlider(Qt.Orientation.Horizontal)
    explorer.threshold_slider.setRange(0, 100)
    explorer.threshold_slider.setValue(40)
    explorer.threshold_label = QLabel("40 %")
    explorer.threshold_slider.valueChanged.connect(
        lambda slider_value: explorer.threshold_label.setText(f"{slider_value} %")
    )
    th_row = QHBoxLayout()
    th_row.addWidget(explorer.threshold_slider)
    th_row.addWidget(explorer.threshold_label)
    form_layout.addRow("Threshold", th_row)

    explorer.beta_spin = QDoubleSpinBox()
    explorer.beta_spin.setRange(1.0, 100.0)
    explorer.beta_spin.setValue(30.0)
    explorer.beta_spin.setDecimals(1)
    form_layout.addRow("Sigmoid β", explorer.beta_spin)

    pos_lbl = QLabel("Phrases: direction to KEEP (higher match)")
    pos_lbl.setStyleSheet("font-weight: normal; text-transform: none; color: #afa;")
    form_layout.addRow(pos_lbl)
    explorer.prefilter_pos_edit = QPlainTextEdit()
    explorer.prefilter_pos_edit.setPlaceholderText("One phrase per line…")
    explorer.prefilter_pos_edit.setMinimumHeight(88)
    explorer.prefilter_pos_edit.setPlainText("\n".join(CLIP_PREFILTER_DEFAULT_PROMPTS["pos"]))
    form_layout.addRow(explorer.prefilter_pos_edit)

    neg_lbl = QLabel("Phrases: contrast / away from (lower → reject)")
    neg_lbl.setStyleSheet("font-weight: normal; text-transform: none; color: #faa;")
    form_layout.addRow(neg_lbl)
    explorer.prefilter_neg_edit = QPlainTextEdit()
    explorer.prefilter_neg_edit.setPlaceholderText("One phrase per line…")
    explorer.prefilter_neg_edit.setMinimumHeight(88)
    explorer.prefilter_neg_edit.setPlainText("\n".join(CLIP_PREFILTER_DEFAULT_PROMPTS["neg"]))
    form_layout.addRow(explorer.prefilter_neg_edit)

    prefilter_panel.main_layout.addLayout(form_layout)
    reset_prompts_row = QHBoxLayout()
    btn_reset_pf_prompts = QPushButton("Reset phrases to defaults")
    btn_reset_pf_prompts.clicked.connect(explorer.reset_clip_prefilter_prompts)
    reset_prompts_row.addWidget(btn_reset_pf_prompts)
    reset_prompts_row.addStretch()
    prefilter_panel.main_layout.addLayout(reset_prompts_row)
    prefilter_panel.main_layout.addStretch()
    btn4 = QPushButton("PREFILTER (CLIP) ◆")
    btn4.setObjectName("RunBtn")
    btn4.clicked.connect(explorer.on_run_prefilter)
    prefilter_panel.main_layout.addWidget(btn4)
    page_layout.addWidget(prefilter_panel)
    explorer.stacked_tools.addWidget(prefilter_page)
    explorer.run_buttons.append(btn4)
