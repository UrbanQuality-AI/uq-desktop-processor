"""
Connects pipeline execution and progress signals to the explorer UI.
"""

import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from uq_desktop_processor.evaluation.clip_prefilter.defaults import FILTER_PROMPTS as CLIP_PREFILTER_DEFAULT_PROMPTS
from uq_desktop_processor.gui.shell.formatters import (
    format_download_summary,
    format_euler_routes_summary,
    format_prefilter_summary,
    format_vit_summary,
)
from uq_desktop_processor.gui.shell.worker import StepThread
from uq_desktop_processor.street_view_analysis import EulerRoutesResult, generate_clean_routes

if TYPE_CHECKING:
    Base = QWidget
else:
    Base = object


class PipelineMixin(Base):
    """
    PipelineMixin UI helper class.
    """

    prefilter_pos_edit: QPlainTextEdit
    prefilter_neg_edit: QPlainTextEdit
    model_combo: QComboBox
    planner_source_combo: QComboBox
    place_name_edit: QLineEdit
    region_path_edit: QLineEdit
    road_path_edit: QLineEdit
    spacing_edit: QLineEdit
    min_dist_edit: QLineEdit
    planner_points_out_edit: QLineEdit
    token_edit: QLineEdit
    mapillary_points_edit: QLineEdit
    mapillary_out_edit: QLineEdit
    radius_edit: QLineEdit
    workers_spin: QSpinBox
    device_combo: QComboBox
    prefilter_image_folder_edit: QLineEdit
    prefilter_rejected_edit: QLineEdit
    threshold_slider: QSlider
    beta_spin: QDoubleSpinBox
    vit_images_dir_edit: QLineEdit
    vit_model_path_edit: QLineEdit
    vit_calibrators_edit: QLineEdit
    vit_output_path_edit: QLineEdit
    vit_output_layer_name_edit: QLineEdit
    vit_model_name_edit: QLineEdit
    vit_image_size_spin: QSpinBox
    vit_batch_size_spin: QSpinBox
    vit_device_combo: QComboBox
    progress_bar: QProgressBar
    run_buttons: Iterable[QPushButton]
    route_source_combo: QComboBox
    route_grid_cols: QSpinBox
    route_grid_rows: QSpinBox
    route_out_edit: QLineEdit
    route_consolidate_tol: QDoubleSpinBox
    route_cache_check: QCheckBox
    route_city_edit: QLineEdit
    route_region_path_edit: QLineEdit
    route_roads_path_edit: QLineEdit

    pipeline: Any
    map_canvas: Any
    _worker: StepThread | None
    _last_euler_result: EulerRoutesResult | None

    def _append_data_science_section(self, title: str, lines: list[str]) -> None:
        """
        Run  append data science section.

        :param title: See caller/context.
        :param lines: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _append_data_science_section(title, lines)
            Out: UI/application state updated as intended.
        """
        pass

    def reset_clip_prefilter_prompts(self) -> None:
        """
        Run reset clip prefilter prompts.

        :return: Result of this step or updated UI/application state.

        Example::
            In: reset_clip_prefilter_prompts()
            Out: UI/application state updated as intended.
        """
        self.prefilter_pos_edit.setPlainText("\n".join(CLIP_PREFILTER_DEFAULT_PROMPTS["pos"]))
        self.prefilter_neg_edit.setPlainText("\n".join(CLIP_PREFILTER_DEFAULT_PROMPTS["neg"]))

    def _apply_config_from_ui(self) -> None:
        """
        Run  apply config from ui.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _apply_config_from_ui()
            Out: UI/application state updated as intended.
        """
        model_tuple = self.model_combo.currentData()
        if not isinstance(model_tuple, tuple):
            # Keep a valid fallback if combo-box payload is malformed.
            model_tuple = ("ViT-L/14@336px",)

        # Snapshot all controls into pipeline config before launching any step.
        self.pipeline.update_config(
            {
                "planner_source": self.planner_source_combo.currentData(),
                "place_name": self.place_name_edit.text().strip(),
                "region_geojson_path": self.region_path_edit.text().strip(),
                "road_geojson_path": self.road_path_edit.text().strip(),
                "spacing": float(self.spacing_edit.text() or "100"),
                "min_distance": float(self.min_dist_edit.text() or "50"),
                "points_layer_path": self.planner_points_out_edit.text().strip(),
                "mapillary_token": self.token_edit.text().strip() or os.environ.get("MAPILLARY_ACCESS_TOKEN", ""),
                "mapillary_points_path": self.mapillary_points_edit.text().strip(),
                "mapillary_images_output_dir": self.mapillary_out_edit.text().strip(),
                "search_radius": float(self.radius_edit.text() or "150"),
                "mapillary_max_workers": int(self.workers_spin.value()),
                "clip_model_names": model_tuple,
                "torch_device": self.device_combo.currentData(),
                "prefilter_image_folder": self.prefilter_image_folder_edit.text().strip(),
                "prefilter_rejected_folder": self.prefilter_rejected_edit.text().strip(),
                "prefilter_pos_prompts": self.prefilter_pos_edit.toPlainText(),
                "prefilter_neg_prompts": self.prefilter_neg_edit.toPlainText(),
                "filter_threshold": float(self.threshold_slider.value()),
                "beta_sigmoid": float(self.beta_spin.value()),
                "vit_images_dir": self.vit_images_dir_edit.text().strip(),
                "vit_model_path": self.vit_model_path_edit.text().strip(),
                "vit_calibrators_dir": self.vit_calibrators_edit.text().strip(),
                "vit_output_layer_path": self.vit_output_path_edit.text().strip(),
                "vit_output_layer_name": self.vit_output_layer_name_edit.text().strip(),
                "vit_model_name": self.vit_model_name_edit.text().strip(),
                "vit_image_size": int(self.vit_image_size_spin.value()),
                "vit_batch_size": int(self.vit_batch_size_spin.value()),
                "vit_torch_device": self.vit_device_combo.currentData(),
            }
        )
        self.pipeline.base_dir = Path(self.pipeline.config["base_dir"])
        image_output_dir = (self.pipeline.config.get("mapillary_images_output_dir") or "").strip()
        if image_output_dir:
            self.pipeline.raw_dir = Path(image_output_dir).expanduser().resolve()
        else:
            # Planner default output lives under base_dir/images/raw.
            self.pipeline.raw_dir = self.pipeline.base_dir / "images" / "raw"
        self.pipeline.rejected_dir = self.pipeline.base_dir / "images" / "rejected"
        self.pipeline.results_dir = self.pipeline.base_dir / "results"
        self.pipeline.ensure_directories()

    def _set_busy(self, busy: bool) -> None:
        """
        Run  set busy.

        :param busy: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _set_busy(busy)
            Out: UI/application state updated as intended.
        """
        self.progress_bar.setRange(0, 0 if busy else 100)
        if not busy:
            self.progress_bar.setValue(0)
        for run_button in self.run_buttons:
            run_button.setDisabled(busy)

    def _run_async(self, task_fn: Callable[[], Any], on_ok: Callable[[Any], None] | None = None) -> None:
        """
        Run  run async.

        :param task_fn: See caller/context.
        :param on_ok: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _run_async(task_fn, on_ok)
            Out: UI/application state updated as intended.
        """
        if self._worker is not None and self._worker.isRunning():
            logging.getLogger(__name__).warning("An operation is already running.")
            return
        self._apply_config_from_ui()
        self._set_busy(True)

        self._worker = StepThread(task_fn, self)

        def _done(result: Any, error_message: str) -> None:
            """
            Run  done.

            :param result: See caller/context.
            :param error_message: See caller/context.
            :return: Result of this step or updated UI/application state.

            Example::
                In: _done(result, error_message)
                Out: UI/application state updated as intended.
            """
            self._set_busy(False)
            if error_message:
                logging.getLogger(__name__).error("Worker thread: %s", error_message)
            elif on_ok is not None:
                # Marshal callbacks back onto the GUI event loop.
                QTimer.singleShot(0, lambda: on_ok(result))
            self._worker = None

        self._worker.finished.connect(_done)
        self._worker.start()

    def _refresh_map_after_points(self) -> None:
        """
        Run  refresh map after points.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _refresh_map_after_points()
            Out: UI/application state updated as intended.
        """
        self.map_canvas.update_planner_layers(self.pipeline.roads_gdf, self.pipeline.points)

    def _refresh_map_after_eval(self) -> None:
        """
        Run  refresh map after eval.

        :return: Result of this step or updated UI/application state.

        Example::
            In: _refresh_map_after_eval()
            Out: UI/application state updated as intended.
        """
        self.map_canvas.update_eval_layer(self.pipeline.roads_gdf, self.pipeline.evaluation_results)
        evaluation_results = self.pipeline.evaluation_results
        if evaluation_results:
            self._append_data_science_section(
                "ViT scoring (fine-tuned)",
                format_vit_summary(
                    evaluation_results,
                    pipeline_config=self.pipeline.config,
                    default_results_dir=self.pipeline.results_dir,
                ),
            )

    def _on_sampling_points_done(self, count: Any) -> None:
        """
        Run  on sampling points done.

        :param count: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _on_sampling_points_done(count)
            Out: UI/application state updated as intended.
        """
        self._refresh_map_after_points()
        out = (self.pipeline.config.get("points_layer_path") or "").strip()
        path_disp = str(Path(out).resolve()) if out else "(unknown)"
        generated_count = int(count) if count is not None else 0
        self._append_data_science_section(
            "Sampling points",
            [
                f"Generated points: {generated_count}",
                f"Output path: {path_disp}",
            ],
        )

    def on_run_planner(self) -> None:
        """
        Run on run planner.

        :return: Result of this step or updated UI/application state.

        Example::
            In: on_run_planner()
            Out: UI/application state updated as intended.
        """
        self._run_async(
            self.pipeline.step_1_generate_points,
            on_ok=self._on_sampling_points_done,
        )

    def _on_download_done(self, stats: Any) -> None:
        """
        Run  on download done.

        :param stats: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _on_download_done(stats)
            Out: UI/application state updated as intended.
        """
        self._refresh_map_after_points()
        if isinstance(stats, dict):
            self._append_data_science_section("Mapillary download", format_download_summary(stats))

    def on_run_download(self) -> None:
        """
        Run on run download.

        :return: Result of this step or updated UI/application state.

        Example::
            In: on_run_download()
            Out: UI/application state updated as intended.
        """
        self._run_async(
            self.pipeline.step_2_download_images,
            on_ok=self._on_download_done,
        )

    def _on_prefilter_done(self, stats: Any) -> None:
        """
        Run  on prefilter done.

        :param stats: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _on_prefilter_done(stats)
            Out: UI/application state updated as intended.
        """
        self._refresh_map_after_points()
        if isinstance(stats, dict):
            self._append_data_science_section("CLIP prefilter", format_prefilter_summary(stats))

    def on_run_prefilter(self) -> None:
        """
        Run on run prefilter.

        :return: Result of this step or updated UI/application state.

        Example::
            In: on_run_prefilter()
            Out: UI/application state updated as intended.
        """
        self._run_async(
            self.pipeline.step_3_prefilter,
            on_ok=self._on_prefilter_done,
        )

    def on_run_vit_evaluate(self) -> None:
        """
        Run on run vit evaluate.

        :return: Result of this step or updated UI/application state.

        Example::
            In: on_run_vit_evaluate()
            Out: UI/application state updated as intended.
        """
        self._run_async(
            self.pipeline.step_6_evaluate_vit,
            on_ok=lambda _result: self._refresh_map_after_eval(),
        )

    @staticmethod
    def on_run_export(_checked: bool = False) -> None:
        """
        Run on run export.

        :param _checked: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: on_run_export(_checked)
            Out: UI/application state updated as intended.
        """
        logging.getLogger(__name__).warning("Export UI is not enabled yet. This tab is intentionally empty for now.")

    def on_run_euler_routes(self) -> None:
        """
        Run on run euler routes.

        :return: Result of this step or updated UI/application state.

        Example::
            In: on_run_euler_routes()
            Out: UI/application state updated as intended.
        """
        route_source = self.route_source_combo.currentData()
        kwargs: dict[str, Any] = {
            "grid_size": (int(self.route_grid_cols.value()), int(self.route_grid_rows.value())),
            "output_dir": self.route_out_edit.text().strip() or "routes_chinese_postman_gpx",
            "consolidate_tolerance_m": float(self.route_consolidate_tol.value()),
            "use_cache": self.route_cache_check.isChecked(),
        }
        if route_source == "place":
            city = self.route_city_edit.text().strip()
            if not city:
                logging.getLogger(__name__).warning("Enter a city / place name.")
                return
            kwargs["city_name"] = city
            kwargs["region_geojson_path"] = None
            kwargs["road_geojson_path"] = None
        elif route_source == "region":
            region_path = self.route_region_path_edit.text().strip()
            if not region_path:
                logging.getLogger(__name__).warning("Select a region polygon GeoJSON file.")
                return
            kwargs["region_geojson_path"] = region_path
            kwargs["city_name"] = None
            kwargs["road_geojson_path"] = None
        elif route_source == "roads":
            roads_path = self.route_roads_path_edit.text().strip()
            if not roads_path:
                logging.getLogger(__name__).warning("Select a road network GeoJSON file.")
                return
            kwargs["road_geojson_path"] = roads_path
            kwargs["city_name"] = None
            kwargs["region_geojson_path"] = None
        else:
            logging.getLogger(__name__).error("Unknown route source: %s", route_source)
            return

        def task() -> EulerRoutesResult:
            """
            Run task.

            :return: Result of this step or updated UI/application state.

            Example::
                In: task()
                Out: UI/application state updated as intended.
            """
            return generate_clean_routes(**kwargs)

        self._run_async(task, on_ok=self._on_euler_routes_done)

    def _on_euler_routes_done(self, result: Any) -> None:
        """
        Run  on euler routes done.

        :param result: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _on_euler_routes_done(result)
            Out: UI/application state updated as intended.
        """
        if not isinstance(result, EulerRoutesResult):
            return
        self._last_euler_result = result
        polylines = [list(polyline) for polyline in result.polylines_wgs84]
        self.map_canvas.update_euler_layer(polylines)
        self._append_data_science_section(
            "Chinese postman / GPX drive routes",
            format_euler_routes_summary(result),
        )
