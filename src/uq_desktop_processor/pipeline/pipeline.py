"""
Orchestrates end-to-end processing: points, images, prefilter, CLIP eval, vector export.
"""

import logging
import os
from pathlib import Path
from typing import Any

from uq_desktop_processor.evaluation import (
    evaluate_images_with_clip,
    prefilter_folder,
)
from uq_desktop_processor.evaluation.finetuned_evaluator.run import evaluate_images_with_finetuned
from uq_desktop_processor.layer_creation import export_point_layer
from uq_desktop_processor.pipeline.defaults import DEFAULT_CONFIG
from uq_desktop_processor.pipeline.points_io import load_sampling_points_latlon
from uq_desktop_processor.pipeline.prefilter_config import prefilter_prompts_from_config
from uq_desktop_processor.street_view_analysis import build_points_pipeline, download_mapillary_images


class UrbanQualityAIPipeline:
    """
    Application pipeline controller.

    Holds in-memory state (roads, points, evaluation) and exposes steps to run sequentially.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize pipeline state and merge runtime overrides with defaults.

        :param config: Optional dictionary with config overrides.

        Example::
            In: UrbanQualityAIPipeline({"base_dir": "data_dev"})
            Out: pipeline object with directories initialized under "data_dev"
        """
        self.log = logging.getLogger("UrbanQualityAI")
        self.log.setLevel(logging.DEBUG)

        self.config = {**DEFAULT_CONFIG, **(config or {})}

        self.base_dir = Path(self.config["base_dir"])
        self.raw_dir = self.base_dir / "images" / "raw"
        self.rejected_dir = self.base_dir / "images" / "rejected"
        self.results_dir = self.base_dir / "results"

        self.roads_gdf = None
        self.points: list[tuple[float, float]] = []
        self.evaluation_results = None

        self._ensure_directories()

    def _resolve_torch_device(self) -> str:
        """
        Resolve torch device from config and environment fallback rules.

        :return: ``"cpu"`` or ``"cuda"``.

        Example::
            In: pipeline._resolve_torch_device()
            Out: "cpu"
        """
        configured_device = self.config.get("torch_device", "auto")
        if configured_device == "cpu":
            return "cpu"
        if configured_device == "cuda":
            return "cuda"
        return "cuda" if os.environ.get("FORCE_CUDA") == "1" else "cpu"

    def update_config(self, new_config: dict[str, Any]) -> None:
        """
        Merge new settings into active runtime config.

        :param new_config: Partial config dictionary to apply.

        Example::
            In: pipeline.update_config({"spacing": 80})
            Out: pipeline.config["spacing"] == 80
        """
        self.config.update(new_config)
        self.log.info("Configuration updated: %s", new_config)

    def _ensure_directories(self) -> None:
        """Create raw/rejected/results directories if they are missing."""
        for directory_path in [self.raw_dir, self.rejected_dir, self.results_dir]:
            directory_path.mkdir(parents=True, exist_ok=True)

    def ensure_directories(self) -> None:
        """
        Public wrapper for directory bootstrap.

        Example::
            In: pipeline.ensure_directories()
            Out: required directories exist on disk
        """
        self._ensure_directories()

    def step_1_generate_points(self) -> int:
        """
        Step 1: generate sampling points and save them to the configured layer path.

        :return: Number of generated points.

        Example::
            In: pipeline.step_1_generate_points()
            Out: 1248
        """
        planner_source = self.config.get("planner_source", "place")
        spacing = float(self.config["spacing"])
        min_distance = float(self.config["min_distance"])
        points_output_path = (self.config.get("points_layer_path") or "").strip()
        if not points_output_path:
            raise ValueError("Set ``points_layer_path`` to a .geojson / .gpkg output path.")

        self.log.info("=== STEP 1: Point generation (source=%s, spacing=%sm) ===", planner_source, spacing)

        try:
            # Choose exactly one planner source mode.
            if planner_source == "place":
                place = self.config["place_name"]
                self.roads_gdf, self.points = build_points_pipeline(
                    output_points_path=points_output_path,
                    place_name=place,
                    spacing=spacing,
                    min_distance_m=min_distance,
                )
            elif planner_source == "region":
                region_path = (self.config.get("region_geojson_path") or "").strip()
                if not region_path:
                    raise ValueError("Region GeoJSON path is empty.")
                self.roads_gdf, self.points = build_points_pipeline(
                    output_points_path=points_output_path,
                    region_geojson_path=region_path,
                    spacing=spacing,
                    min_distance_m=min_distance,
                )
            elif planner_source == "roads":
                roads_path = (self.config.get("road_geojson_path") or "").strip()
                if not roads_path:
                    raise ValueError("Road network GeoJSON path is empty.")
                self.roads_gdf, self.points = build_points_pipeline(
                    output_points_path=points_output_path,
                    road_geojson_path=roads_path,
                    spacing=spacing,
                    min_distance_m=min_distance,
                )
            else:
                raise ValueError(f"Unknown planner source: {planner_source}")

            count = len(self.points)
            self.log.info("Success. Generated %s points; saved to %s", count, Path(points_output_path).resolve())
            return count
        except Exception as error:
            self.log.error("Step 1 failed: %s", error)
            raise

    def step_2_download_images(self) -> dict[str, Any]:
        """
        Step 2: download Mapillary images for loaded sampling points.

        :return: Download summary dictionary.

        Example::
            In: pipeline.step_2_download_images()["downloaded"]
            Out: 915
        """
        self.log.info("=== STEP 2: Mapillary download ===")

        points_path = (self.config.get("mapillary_points_path") or "").strip()
        if not points_path:
            points_path = (self.config.get("points_layer_path") or "").strip()
        if not points_path:
            raise ValueError("Set a point layer path (Mapillary tab or planner output path).")
        try:
            # Reload points from disk to keep GUI/CLI behavior consistent.
            self.points = load_sampling_points_latlon(points_path)
        except FileNotFoundError as error:
            self.log.error("%s", error)
            raise RuntimeError("Sampling points file missing. Run step 1 (generate points) first.") from error
        if not self.points:
            raise RuntimeError(f"Point layer is empty: {points_path}")

        token = self.config["mapillary_token"]
        if not token:
            self.log.error("Mapillary token is missing.")
            raise ValueError("A Mapillary access token is required.")

        image_output_dir = (self.config.get("mapillary_images_output_dir") or "").strip()
        if image_output_dir:
            self.raw_dir = Path(image_output_dir).expanduser().resolve()
        else:
            self.raw_dir = self.base_dir / "images" / "raw"
        self._ensure_directories()

        # Downloader returns per-point outcomes plus aggregate timing/count metrics.
        stats = download_mapillary_images(
            points=self.points,
            output_folder=str(self.raw_dir),
            mapillary_token=token,
            search_radius_m=float(self.config["search_radius"]),
            fov_deg=float(self.config.get("mapillary_fov_deg", 90)),
            max_workers=int(self.config.get("mapillary_max_workers", 20)),
        )
        self.log.info("Download finished.")
        return stats

    def step_3_prefilter(self) -> dict[str, Any]:
        """
        Step 3: run CLIP prefilter and split images into kept/rejected groups.

        :return: Prefilter statistics dictionary.

        Example::
            ``pipeline.step_3_prefilter()["summary"]["kept"] -> 640``
        """
        self.log.info("=== STEP 3: CLIP prefilter ===")

        device = self._resolve_torch_device()

        configured_image_folder = (self.config.get("prefilter_image_folder") or "").strip()
        if not configured_image_folder:
            image_folder = str(self.raw_dir.resolve())
        else:
            image_folder = str(Path(configured_image_folder).expanduser().resolve())

        rejected_spec = (self.config.get("prefilter_rejected_folder") or "").strip()
        if not rejected_spec:
            # Relative name means "inside image_folder".
            rejected_spec = "rejected"

        model_names = tuple(self.config.get("clip_model_names") or ("ViT-L/14@336px",))
        stats = prefilter_folder(
            image_folder=image_folder,
            rejected_folder=rejected_spec,
            device=device,
            model_names=model_names,
            beta_sigmoid=float(self.config.get("beta_sigmoid", 30.0)),
            filter_threshold=float(self.config.get("filter_threshold", 40.0)),
            filter_prompts=prefilter_prompts_from_config(self.config),
        )

        kept_count = stats["summary"]["kept"]
        rejected_count = stats["summary"]["rejected"]
        self.log.info("Prefilter done. Kept: %s, rejected: %s", kept_count, rejected_count)
        return stats

    def step_4_evaluate(self) -> dict[str, Any]:
        """
        Step 4: run CLIP scoring for configured dimensions.

        :return: Evaluation results dictionary.

        Example::
            In: pipeline.step_4_evaluate()
            Out: {"results": [...], "summary": {...}, ...}
        """
        self.log.info("=== STEP 4: CLIP image scoring ===")

        device = self._resolve_torch_device()

        model_names = tuple(self.config.get("clip_model_names") or ("ViT-L/14@336px",))
        evaluation_results = evaluate_images_with_clip(
            image_folder=str(self.raw_dir),
            device=device,
            model_names=model_names,
            beta_sigmoid=float(self.config.get("beta_sigmoid", 30.0)),
            raise_on_validation_error=True,
        )
        self.evaluation_results = evaluation_results

        self.log.info("Scoring finished successfully.")
        return evaluation_results

    def step_5_export(self) -> str:
        """
        Step 5: export scored points to a vector file.

        :return: Path to saved output file.

        Example::
            In: pipeline.step_5_export()
            Out: "data/results/urban_quality_ai_output.geojson"
        """
        self.log.info("=== STEP 5: Export results ===")

        if not self.evaluation_results:
            self.log.warning("No evaluation results. Run step 4 first.")
            raise RuntimeError("Nothing to export.")

        custom = (self.config.get("export_path") or "").strip()
        if custom:
            output_path = Path(custom)
            if not output_path.is_absolute():
                # Keep relative paths anchored to current working directory.
                output_path = Path.cwd() / output_path
        else:
            output_path = self.results_dir / "urban_quality_ai_output.geojson"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path = export_point_layer(
            results=self.evaluation_results,
            output_path=str(output_path),
        )
        self.log.info("Written to %s", saved_path)
        return str(saved_path)

    def step_6_evaluate_vit(self) -> dict[str, Any]:
        """
        Step 6: run fine-tuned ViT scoring and export result layer.

        :return: Evaluation results dictionary.

        Example::
            In: pipeline.step_6_evaluate_vit()
            Out: {"results": [...], "summary": {...}, ...}
        """
        self.log.info("=== STEP 6: Fine-tuned ViT scoring ===")

        images_dir_spec = (self.config.get("vit_images_dir") or "").strip()
        if not images_dir_spec:
            raise ValueError("Set 'vit_images_dir' (folder with images).")
        images_dir = str(Path(images_dir_spec).expanduser().resolve())

        model_path_spec = (self.config.get("vit_model_path") or "").strip()
        if not model_path_spec:
            # Backward-compatible alias used by some configs.
            model_path_spec = (self.config.get("vit_checkpoint_path") or "").strip()
        if not model_path_spec:
            raise ValueError("Set 'vit_model_path' (model weights file).")
        model_path = Path(model_path_spec)

        calibrators_dir_spec = (self.config.get("vit_calibrators_dir") or "").strip()
        calibrators_dir = Path(calibrators_dir_spec) if calibrators_dir_spec else None

        output_layer_spec = (self.config.get("vit_output_layer_path") or "").strip()
        output_layer_path = (
            Path(output_layer_spec) if output_layer_spec else (self.results_dir / "vit_finetuned_scores.gpkg")
        )
        output_layer_name = (self.config.get("vit_output_layer_name") or "").strip() or None

        model_name = str(self.config.get("vit_model_name") or "vit_base_patch14_dinov2.lvd142m")
        image_size = int(self.config.get("vit_image_size") or 224)
        batch_size = int(self.config.get("vit_batch_size") or 32)
        vit_torch_device = str(self.config.get("vit_torch_device") or "auto")

        evaluation_results = evaluate_images_with_finetuned(
            images_dir=images_dir,
            model_path=model_path,
            calibrators_dir=calibrators_dir,
            model_name=model_name,
            image_size=image_size,
            batch_size=batch_size,
            torch_device=vit_torch_device,
        )
        self.evaluation_results = evaluation_results

        # Export is intentionally done at the pipeline layer (avoid evaluation -> layer_creation coupling).
        output_layer_path.parent.mkdir(parents=True, exist_ok=True)
        export_point_layer(
            results=evaluation_results,
            output_path=str(output_layer_path),
            layer=output_layer_name,
        )
        self.log.info("Exported fine-tuned results to: %s", output_layer_path)

        self.log.info("ViT scoring finished successfully.")
        return evaluation_results
