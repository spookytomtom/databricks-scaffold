# Agent Guidelines for databricks-scaffold

## Critical: `pyspark` is NOT a Required Dependency

**Do NOT add `pyspark` to `install_requires`, `requirements.txt`, or any dependency manifest.**

### Why `pyspark` Must Not Be Declared as a Dependency

1. **Databricks notebooks (browser):** PySpark is pre-installed and pre-configured by the Databricks Runtime. It is injected into the notebook globals automatically. Installing a separate `pyspark` package is unnecessary and can cause version conflicts with the Runtime's bundled Spark.

2. **Databricks Connect (local VS Code / IDE):** The entire purpose of Databricks Connect is that **PySpark code runs remotely in the cloud**. The local machine acts as the cluster driver — it submits jobs to the cluster, but the actual Spark execution happens there. The `databricks-connect` package bundles its own compatible PySpark client. Installing a separate `pyspark` alongside `databricks-connect` causes hard-to-debug version conflicts and is explicitly discouraged by Databricks.

3. **Polars runs locally:** This library uses Polars for single-node DataFrame processing on the driver (local machine). Polars is the only heavy local dependency. The Spark ↔ Polars bridge works through Parquet files written to a Unity Catalog Volume (or local temp), not through in-memory PySpark collections.

### What This Means for Development

- **Never `import pyspark` at the top level** of any module that might be imported before a Spark session exists. Use lazy imports inside functions/methods where Spark objects are actually needed.
- **Type hints:** Use string forward references (e.g., `"pyspark.sql.DataFrame"`) or `typing.TYPE_CHECKING` guards to avoid importing pyspark at import time.
- **Testing:** Mock Spark objects; do not require a live Spark session or installed `pyspark` to run unit tests.
- **Packaging:** `setup.py` / `pyproject.toml` should list `polars` and other local utilities, but **never `pyspark`**.

### Recommended Session Setup (On-Cluster and Local)

```python
try:
    spark  # Already injected on a Databricks cluster
except NameError:
    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.getOrCreate()
```

This pattern works identically in both environments without any code changes.
