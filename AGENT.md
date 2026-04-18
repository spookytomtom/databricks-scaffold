# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Docs Search

When you need to look up documentation for PySpark, Polars, Databricks, or related libraries, use the `context7` MCP tool to fetch up-to-date docs.

## Commands

Install the package in editable mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

Run all tests:
```bash
pytest tests/
```

Run a single test:
```bash
pytest tests/test_core.py::test_xxx
```

Run tests with coverage:
```bash
pytest tests/ --cov=databricks_scaffold
```

Build and publish:
```bash
python -m build && twine upload dist/*
```

## Import Convention

Import from the package root: `from databricks_scaffold import ...`  
**Not** `from src.databricks_scaffold import ...`

## Architecture

The library bridges PySpark and Polars by writing Parquet to Unity Catalog Volumes or local driver `/tmp`, then reading back with the other engine. Installed in notebooks via `%pip install git+https://github.com/spookytomtom/databricks-scaffold.git`.

**Modules:**

- **`_internal.py`**: Shared helpers used by both `core.py` and `utils.py`.
  - `_get_notebook_var()`: Reads a variable from the IPython notebook namespace.
  - `_resolve_is_dev()`: Three-tier resolution — explicit arg → notebook `IS_DEV` → default `True`. Handles Databricks widget strings (`"False"`, `"0"`, etc.).

- **`core.py` — `VolumeSpiller`**: Central class managing the Parquet spill buffer.
  - `spark_to_polars()` / `polars_to_spark()`: Write Parquet to volume, read back with other engine. Temp dirs tracked in `_active_volume_dirs`, cleaned on `teardown()`. Note: `polars_to_spark()` is volume-only (no `local` option).
  - `save_checkpoint_pl` / `load_checkpoint_pl`: Support both `volume` and `local` storage. `save_checkpoint_spark` / `load_checkpoint_spark` are volume-only.
  - `teardown()`: Clears local `/tmp` driver storage tracked in `_active_local_dirs`. Volume cleanup is conditional on `IS_DEV` (preserved in dev, dropped in prod).
  - `_prepare_polars_timestamps()`: Auto-casts Polars `ns`/`us` Datetime to `ms` + UTC before writing.
  - Checkpoint names validated with `^[\w\-]+$` (underscores/hyphens only — no slashes or dots).

- **`utils.py` — `DataProfiler` + standalone functions**:
  - `DataProfiler.profile()`: Auto-detects Polars vs PySpark, dispatches accordingly.
  - `display2()`: Calls Databricks `display()` only when `IS_DEV` is truthy. Reads from notebook namespace via `_get_notebook_var()`. Skips gracefully in pytest.
  - `glimpse()`, `frame_shape()`, `keep_duplicates()`, `is_unique()`, `clean_column_names()`, `apply_column_comments()`: Spark utility functions.

## IS_DEV

Controls all behavior — set once at the top of every notebook:
- `IS_DEV = True` (default): volume created with `IF NOT EXISTS`; `teardown()` preserves volume data; `display2()` renders.
- `IS_DEV = False` (prod): volume dropped and recreated on init; `teardown()` drops volume; `display2()` is silent.

Both `VolumeSpiller` and `display2()` read `IS_DEV` from the notebook namespace automatically via `_resolve_is_dev()`. Resolution priority: explicit constructor arg → notebook `IS_DEV` variable → `True`. Databricks widget strings (`"False"`, `"0"`, `"no"`) are handled correctly. Override per-instance with `VolumeSpiller(..., is_dev=False)`.

## Testing

Tests run against a local `SparkSession` — no Databricks cluster needed. `conftest.py` monkey-patches `spark.sql` (to `None`) and overrides `volume_root` / `local_base_dir` to `tmp_path`. **These patches are intentional** — do not change them. The `spark` fixture is session-scoped; `spiller` is function-scoped.

## Compression

Hardcoded — do not change without reason:
- Volume path: `zstd`
- Local driver path: `snappy`

## Databricks Connect support

`VolumeSpiller` works under Databricks Connect as well as on-cluster. Detection runs once at `__init__` via `_is_databricks_connect(spark)`. This function checks `isinstance(spark, DatabricksSession)` **and** also checks `type(spark).__module__ == "pyspark.sql.connect.session"` — because `DatabricksSession.builder.getOrCreate()` returns a `pyspark.sql.connect.session.SparkSession`, not a `DatabricksSession` instance. No user action required.

Under Databricks Connect, every operation that would otherwise touch `/Volumes/...` from local Python code is routed through the Databricks SDK Files API (`WorkspaceClient.files`). Polars writes to a local `/tmp` staging dir first, then `files.upload_from` streams the parquet up. For reads, `files.download_to` pulls parquet into `/tmp` before Polars opens it.

- **Detection:** `_is_databricks_connect(spark)` — returns False for plain `SparkSession`.
- **Lazy SDK import:** `WorkspaceClient` is only constructed on first volume I/O under Connect. Users who never leave on-cluster mode don't pay for the import.
- **Transfer helpers:** `_upload_dir_to_volume`, `_download_volume_dir` handle the `/tmp` ↔ volume bridge.
- **Volume-aware primitives:** `_volume_mkdirs`, `_volume_exists`, `_volume_listdir`, `_volume_rmtree` — all no-op switches between `os.*`/`shutil` and the Files API based on `self._is_connect`.

If you modify `core.py`: any new code path that touches `/Volumes/...` from Python (not from a Spark API call) MUST go through a `_volume_*` helper. A plain `os.makedirs`, `os.path.exists`, `shutil.rmtree`, `pl.read_parquet`, or `df.write_parquet` against `self.volume_root` silently breaks Connect support.
