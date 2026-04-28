"""
Small string formatting helpers for pipeline status and shell UI labels.
"""

from pathlib import Path
from typing import Any

import numpy as np

from uq_desktop_processor.gui.map_view.geo import polyline_geodesic_length_m
from uq_desktop_processor.street_view_analysis import EulerRoutesResult


def format_euler_routes_summary(result: EulerRoutesResult) -> list[str]:
    """
    Run format euler routes summary.

    :param result: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: format_euler_routes_summary(result)
        Out: UI/application state updated as intended.
    """
    file_count = len(result.gpx_paths)
    polylines = result.polylines_wgs84
    segment_lengths_m = [polyline_geodesic_length_m(polyline) for polyline in polylines]
    total_length_m = sum(segment_lengths_m)
    lines: list[str] = [
        f"Output directory: {result.output_dir}",
        f"GPX files written: {file_count}",
        f"Route segments (polylines): {len(polylines)}",
    ]
    for segment_index, segment_length_m in enumerate(segment_lengths_m, start=1):
        lines.append(f"  Segment {segment_index}: {segment_length_m / 1000.0:.3f} km")
    lines.append(f"Total length (all segments): {total_length_m / 1000.0:.3f} km")
    return lines


def format_download_summary(stats: dict[str, Any]) -> list[str]:
    """
    Run format download summary.

    :param stats: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: format_download_summary(stats)
        Out: UI/application state updated as intended.
    """
    point_count = int(stats.get("point_count") or 0)
    downloaded_count = int(stats.get("downloaded") or 0)
    elapsed = float(stats.get("elapsed_s") or 0.0)
    output_folder = str(stats.get("output_folder") or "")
    images_per_second = float(stats.get("images_per_second") or 0.0)
    failed_messages: list[str] = list(stats.get("failed_messages") or [])
    lines = [
        f"Points processed: {point_count}",
        f"Images saved: {downloaded_count}",
        f"Output path: {output_folder}",
        f"Elapsed: {elapsed:.1f} s",
        f"Mean throughput (saved images / s): {images_per_second:.2f}",
    ]
    if failed_messages:
        max_show = 12
        lines.append(f"Not downloaded / failed ({len(failed_messages)}):")
        for failed_message in failed_messages[:max_show]:
            lines.append(f"  • {failed_message}")
        if len(failed_messages) > max_show:
            lines.append(f"  … and {len(failed_messages) - max_show} more")
    else:
        lines.append("Failures: none")
    return lines


def format_prefilter_summary(stats: dict[str, Any]) -> list[str]:
    """
    Run format prefilter summary.

    :param stats: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: format_prefilter_summary(stats)
        Out: UI/application state updated as intended.
    """
    summary = stats.get("summary") or {}
    rejected_folder = stats.get("rejected_folder") or ""
    rejected_count = int(summary.get("rejected") or 0)
    kept_count = int(summary.get("kept") or 0)
    model_name = stats.get("model_name") or ""
    return [
        f"CLIP model: {model_name}",
        f"Rejected (moved): {rejected_count}",
        f"Kept: {kept_count}",
        f"Rejected folder: {rejected_folder}",
    ]


def format_vit_summary(
    results: dict[str, Any],
    *,
    pipeline_config: dict[str, Any],
    default_results_dir: Path,
) -> list[str]:
    """
    Run format vit summary.

    :param results: See caller/context.
    :param pipeline_config: See caller/context.
    :param default_results_dir: See caller/context.
    :return: Result of this step or updated UI/application state.

    Example::
        In: format_vit_summary(results)
        Out: UI/application state updated as intended.
    """
    out_spec = (pipeline_config.get("vit_output_layer_path") or "").strip()
    if out_spec:
        out_path = str(Path(out_spec).expanduser().resolve())
    else:
        out_path = str((default_results_dir / "vit_finetuned_scores.gpkg").resolve())
    model = str(results.get("model_name") or "")
    images = results.get("images") or []
    order = list(results.get("order") or [])
    checkpoint_path = (pipeline_config.get("vit_model_path") or "").strip()
    checkpoint_line = str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else "(not set)"
    calibrators_spec = (pipeline_config.get("vit_calibrators_dir") or "").strip()
    calibrators_line = str(Path(calibrators_spec).expanduser().resolve()) if calibrators_spec else "(calibration off)"
    arch = (pipeline_config.get("vit_model_name") or "").strip()
    image_size = int(pipeline_config.get("vit_image_size") or 224)
    batch_size = int(pipeline_config.get("vit_batch_size") or 32)
    torch_device = str(pipeline_config.get("vit_torch_device") or "auto")
    lines: list[str] = [
        f"Output path: {out_path}",
        f"Model (architecture): {arch}",
        f"Weights file: {checkpoint_line}",
        f"Calibrators: {calibrators_line}",
        f"Inference: image_size={image_size}, batch_size={batch_size}, device={torch_device}",
        f"Readout model label: {model}",
        f"Images scored: {len(images)}",
    ]
    for cat in order:
        vals: list[float] = []
        for im in images:
            block = (im.get("categories") or {}).get(cat) or {}
            if "probability_pct" in block:
                vals.append(float(block["probability_pct"]))
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        lo = float(arr.min())
        hi = float(arr.max())
        lines.append(f"  {cat}: avg={float(arr.mean()):.2f}%, min={lo:.2f}%, max={hi:.2f}%, range={hi - lo:.2f}%")
    average_overall_pct = results.get("average_overall_pct")
    if average_overall_pct is not None:
        lines.append(f"Overall (mean of per-image means): {float(average_overall_pct):.2f}%")
    skipped_images = results.get("skipped_images") or []
    if skipped_images:
        lines.append(f"Skipped images: {len(skipped_images)}")
    warnings = results.get("warnings") or []
    if warnings:
        lines.append("Warnings: " + "; ".join(str(warning_message) for warning_message in warnings))
    return lines
