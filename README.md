# databricks-scaffold

Productivity utilities for working with Polars on Databricks. Bridges the gap between PySpark and Polars, adds missing DataFrame utilities, and makes single-node workflows on Databricks fast and ergonomic.

```
%pip install git+https://github.com/spookytomtom/databricks-scaffold.git
```

---

## What's included

### `VolumeSpiller`
Uses a Unity Catalog Volume (or local driver `/tmp`) as a Parquet spill buffer to move data between Spark and Polars without collecting through the driver.

| Method | Description |
|--------|-------------|
| `spark_to_polars(df, eager, cleanup, optimize_files)` | Write Spark DF to Volume as Parquet, read back as Polars `DataFrame` or `LazyFrame` |
| `polars_to_spark(df)` | Write Polars DF to Volume as Parquet, read back as Spark `DataFrame` |
| `save_checkpoint_pl(df, name, storage, compression)` | Persist a Polars DF/LazyFrame to a named checkpoint on Volume or local `/tmp` |
| `load_checkpoint_pl(name, eager, storage)` | Load a named Polars checkpoint back as `DataFrame` or `LazyFrame` |
| `save_checkpoint_spark(df, name, optimize_files)` | Persist a Spark DF to a named checkpoint on the Volume |
| `load_checkpoint_spark(name)` | Load a named Spark checkpoint |
| `list_checkpoints(storage)` | List all checkpoint names in Volume or local storage |
| `teardown()` | Delete temp spill dirs and (in prod) drop the Volume |
| `get_path(name)` | Resolve the absolute Volume path for a given name |

**Auto-fixes silently applied on every write:**
- Polars `ns`/`us` `Datetime` columns are cast to `ms` (Spark compatibility)
- Polars `Datetime` columns without a timezone get `UTC` attached (prevents `timestamp_ntz` in Spark)

**`IS_DEV` controls the entire library's behavior — set it once:**
- `IS_DEV = True` — volume created with `IF NOT EXISTS`; `teardown()` preserves data; `display2()` renders output
- `IS_DEV = False` — volume dropped and recreated on init (clean slate); `teardown()` drops it; `display2()` is silent

`VolumeSpiller` and `display2()` both read `IS_DEV` from the notebook namespace automatically. You never need to pass it as an argument. Override per-instance with `VolumeSpiller(..., is_dev=False)` if needed.

---

### `DataProfiler`
Generates a column-level summary (missing values, unique counts, top frequent values) for both Polars and Spark DataFrames.

| Method | Description |
|--------|-------------|
| `profile(df, output="print")` | Auto-detects Polars vs Spark. `output="print"` prints to console; `output="dataframe"` returns a summary DataFrame |

---

### Spark utility functions

| Function | Description |
|----------|-------------|
| `glimpse(df, n=5, truncate=75)` | Vertical schema + data preview, like R's `dplyr::glimpse` |
| `frame_shape(df)` | Prints and returns `(rows, cols)` tuple for a Spark DF |
| `keep_duplicates(df, subset)` | Returns only rows that have duplicates on the given columns. Uses a broadcast join to avoid a full shuffle |
| `is_unique(df, column_name)` | Checks if a column is entirely unique. Short-circuits on first duplicate found |
| `clean_column_names(df)` | Replaces special characters with underscores, collapses runs of underscores, deduplicates collisions — Delta-compatible output |
| `apply_column_comments(spark, table_name, comments, verbose=True)` | Applies column comments to a table via SQL, skipping columns whose comment hasn't changed |
| `display2(df, is_dev=None)` | Calls Databricks `display()` only when `IS_DEV` is truthy. Reads `IS_DEV` from the notebook namespace — same source as `VolumeSpiller` |

---

## Examples

### Setting IS_DEV — do this once at the top of every notebook

```python
# Development: set manually
IS_DEV = True

# Production: Databricks passes it as a job widget
IS_DEV = dbutils.widgets.get("IS_DEV")  # arrives as "False" — parsed automatically
```

That's it. `VolumeSpiller` and `display2()` both read `IS_DEV` from the notebook namespace. No need to pass it anywhere else.

---

### The Polars sandwich: Spark → Polars → Spark

The core use case. Process a distributed Spark DataFrame with Polars on the driver, then push it back.

```python
import polars as pl
from databricks_scaffold import VolumeSpiller

IS_DEV = True  # or from widget

spill = VolumeSpiller(
    spark=spark,
    catalog="main",
    schema="default",
    volume_name="my_spill_vol",
    # is_dev not needed — reads IS_DEV from notebook automatically
)

# Spill to Volume and read as Polars (handles timestamps automatically)
pl_df = spill.spark_to_polars(spark_df, optimize_files=True)

# Standard Polars processing on the driver
result = (
    pl_df
    .filter(pl.col("status") == "active")
    .with_columns(pl.col("revenue").fill_null(0))
    .group_by("region")
    .agg(
        pl.sum("revenue").alias("total_revenue"),
        pl.count("id").alias("customer_count"),
    )
    .sort("total_revenue", descending=True)
)

# Push back to Spark — nanosecond and timezone fixes are applied automatically
final_spark_df = spill.polars_to_spark(result)
final_spark_df.display()

spill.teardown()
```

---

### Streaming large data with LazyFrames

When the data doesn't fit in driver RAM, use `eager=False` to get a `LazyFrame` and stream it directly to a checkpoint without materialising in memory.

```python
# Scan the Volume without loading anything into RAM yet
lazy_pl = spill.spark_to_polars(large_spark_df, eager=False)

# Build a query plan
q = (
    lazy_pl
    .filter(pl.col("event_date") >= "2024-01-01")
    .group_by("user_id")
    .agg(pl.sum("spend").alias("total_spend"))
)

# Sink directly to a checkpoint — never fully materialises on the driver
spill.save_checkpoint_pl(q, name="user_spend_2024", storage="volume")

# Load back later as a LazyFrame for further chaining, or eagerly
df = spill.load_checkpoint_pl("user_spend_2024", eager=True)
```

---

### Checkpointing — stop recalculating expensive steps

```python
# --- First run: expensive computation ---
cleaned_spark = (
    raw_spark_df
    .dropDuplicates(["event_id"])
    .filter("event_date >= '2024-01-01'")
)
spill.save_checkpoint_spark(cleaned_spark, name="cleaned_events")

pl_enriched = (
    spill.spark_to_polars(cleaned_spark)
    .with_columns(pl.col("amount").log1p().alias("log_amount"))
)
spill.save_checkpoint_pl(pl_enriched, name="enriched_events")

# --- Subsequent runs: skip straight to the result ---
pl_enriched = spill.load_checkpoint_pl("enriched_events")

print(spill.list_checkpoints(storage="volume"))
# ['cleaned_events', 'enriched_events']
```

---

### Profiling a DataFrame

Works on both Polars and Spark. Auto-detects the type.

```python
from databricks_scaffold import DataProfiler

profiler = DataProfiler(top_n_freq=5)

# Printed summary
profiler.profile(spark_df)

# Or get a DataFrame back for downstream use
summary_df = profiler.profile(pl_df, output="dataframe")
```

```
=== POLARS DATAFRAME PROFILE ===
Shape: 50000 rows, 6 columns
========================================
Column: region
  Type: String
  Missing: 0 (0.0%)
  Unique: 8 (All Unique: False)
  Top 5: EMEA: 18200 | AMER: 15400 | APAC: 11300 | ...
```

---

### Inspecting Spark DataFrames

```python
from databricks_scaffold import glimpse, frame_shape, is_unique, keep_duplicates

# Vertical schema + sample values
glimpse(df)
# Rows: 50000
# Columns: 5
# $ customer_id   <bigint>  1001, 1002, 1003, 1004, 1005
# $ region        <string>  EMEA, AMER, APAC, EMEA, AMER
# $ revenue       <double>  1200.5, 850.0, null, 3100.0, 200.0
# $ status        <string>  active, churned, active, active, trial
# $ signup_date   <date>    2023-01-10, 2022-07-04, 2023-03-15, ...

# Shape without count + columns separately
frame_shape(df)  # Shape: (50000, 5)

# Column uniqueness check (short-circuits on first duplicate)
is_unique(df, "customer_id")  # Column 'customer_id' is unique: yes

# Keep only duplicate rows for investigation
dupes = keep_duplicates(df, subset=["email", "signup_date"])
```

---

### Cleaning column names for Delta

```python
from databricks_scaffold import clean_column_names

# Input columns: ["Customer ID", "First.Name", "Revenue (USD)", "% Growth"]
df_clean = clean_column_names(df)
# Output columns: ["Customer_ID", "First_Name", "Revenue_USD", "Growth"]
```

---

### Applying column comments efficiently

Only executes SQL for columns whose comment has actually changed.

```python
from databricks_scaffold import apply_column_comments

comments = {
    "customer_id": "Unique identifier for the customer, sourced from CRM",
    "revenue":     "Total billed revenue in USD, excluding refunds",
    "region":      "Sales region: AMER, EMEA, or APAC",
}

apply_column_comments(spark, "main.default.customers", comments)
# Updating 'revenue':
#    Old: 'Billed revenue'
#    New: 'Total billed revenue in USD, excluding refunds'
# Skipping 'region': Input comment is empty.
# --- Done. Updated: 1 | Skipped (No Change): 2 ---
```

---

### Conditional display in notebooks

`display2` reads the same `IS_DEV` variable as `VolumeSpiller` — set it once and both behave correctly.

```python
from databricks_scaffold import display2

IS_DEV = True  # set once at the top of your notebook

# Only renders in dev; silent in prod — no code changes needed between environments
display2(spark_df)
display2(pl_df)        # Polars is converted to pandas automatically
display2(pandas_df)
```

---

### Production pattern

In a Databricks job, define a widget parameter `IS_DEV` with value `False`. The library reads it automatically — no code changes required between dev and prod.

```python
# Job widget sets IS_DEV = "False" — read it once at the top
IS_DEV = dbutils.widgets.get("IS_DEV")

from databricks_scaffold import VolumeSpiller

spill = VolumeSpiller(
    spark=spark,
    catalog="main",
    schema="default",
    volume_name="etl_spill",
    # reads IS_DEV="False" from notebook namespace automatically:
    # - drops + recreates volume on init
    # - teardown() drops it completely
)

try:
    pl_df = spill.spark_to_polars(raw_spark_df, optimize_files=True)
    result = process(pl_df)
    output_spark_df = spill.polars_to_spark(result)
    output_spark_df.write.saveAsTable("main.default.output")
finally:
    spill.teardown()  # volume is dropped, no storage costs left behind
```
