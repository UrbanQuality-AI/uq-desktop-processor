# UQ Desktop Processor

`uq-desktop-processor` is a desktop app and Python package for visual urban assessment from street-level imagery.  
It combines road-network sampling, optional Mapillary downloads (or your own images), CLIP-based prefiltering and scoring, optional ViT workflows, and GIS-friendly export formats.

> Package name: `uq_desktop_processor`  
> Python: `3.12-3.14`

---

## Key Features

- Chinese postman / drive routes with GPX export for efficient field coverage.
- Road-based sampling points with configurable spacing and minimum separation.
- Mapillary image download near points, or processing of local image folders.
- CLIP prefilter that moves low-relevance images to `rejected`.
- CLIP prompt scoring on urban-quality axes (for example: beauty, safety, wealth).
- Optional ViT / finetuned evaluation path.
- Export to GeoJSON, GPKG, SHP, and Parquet via GeoPandas.

---

## Quick Start

### Run with Poetry (recommended)

```bash
poetry install
poetry run uq_desktop_processor
```

### Run with pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
python -m uq_desktop_processor
```

---

## Mapillary Token

For Mapillary downloads, set `MAPILLARY_ACCESS_TOKEN` (or provide the token directly in the GUI).

PowerShell:

```powershell
$env:MAPILLARY_ACCESS_TOKEN = "YOUR_TOKEN_HERE"
```

Bash:

```bash
export MAPILLARY_ACCESS_TOKEN="YOUR_TOKEN_HERE"
```

---

## Typical Data Layout

```text
data/
|- images/
|  |- raw/
|  `- rejected/
`- results/
   |- sampling_points.geojson
   `- urban_quality_ai_output.geojson
```

---

## Documentation and Development

- Build docs locally:

```bash
mkdocs serve
mkdocs build
```

- Common checks:

```bash
poetry run pytest
poetry run mypy src/
poetry run ruff check .
poetry run black --check .
pre-commit run --all-files
```

---

## Project Source

Main package: `src/uq_desktop_processor/`

- `gui/` - PySide6 application shell and pages
- `pipeline/` - pipeline orchestration and defaults
- `evaluation/` - CLIP prefilter/evaluator and finetuned evaluator
- `street_view_analysis/` - sampling, road graph, Mapillary, drive routes
- `layer_creation/` - vector outputs for GIS workflows
