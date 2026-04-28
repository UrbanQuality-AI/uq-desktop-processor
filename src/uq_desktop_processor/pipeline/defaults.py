"""
Default directories and parameters for each pipeline step.
"""

import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "place_name": "Katowice, Poland",
    # Source for sampling-point planning: "place" | "region" | "roads".
    "planner_source": "place",
    "region_geojson_path": "",
    "road_geojson_path": "",
    # Distances are interpreted in meters.
    "spacing": 100,
    "min_distance": 50,
    "search_radius": 100,
    "mapillary_max_workers": 20,
    "mapillary_fov_deg": 90,
    "mapillary_points_path": "",
    "mapillary_images_output_dir": "",
    "base_dir": "data",
    "points_layer_path": str(Path("data") / "results" / "sampling_points.geojson"),
    "mapillary_token": os.environ.get("MAPILLARY_ACCESS_TOKEN", ""),
    "clip_model_names": ("ViT-L/14@336px",),
    "prefilter_image_folder": "",
    "prefilter_rejected_folder": "",
    "prefilter_pos_prompts": "",
    "prefilter_neg_prompts": "",
    "filter_threshold": 40.0,
    "beta_sigmoid": 30.0,
    "export_path": "",
    "torch_device": "auto",
    "vit_images_dir": "",
    "vit_model_path": "",
    "vit_calibrators_dir": "",
    "vit_model_name": "vit_base_patch14_dinov2.lvd142m",
    "vit_image_size": 224,
    "vit_batch_size": 32,
    "vit_torch_device": "auto",
    "vit_output_layer_path": "",
    "vit_output_layer_name": "vit_finetuned_scores",
}
