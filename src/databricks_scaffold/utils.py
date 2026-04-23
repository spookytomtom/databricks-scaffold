import contextlib
import io
import logging
import re
from typing import Any

import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, LongType, StringType, StructField, StructType

from databricks_scaffold._internal import _get_notebook_var, _resolve_is_dev

_logger = logging.getLogger(__name__)


class DataProfiler:
    """
    A utility class for generating summary profiles of Polars and PySpark DataFrames,
    including missing values, unique counts, and top frequent values.
    """

    def __init__(self, top_n: int = 3) -> None:
        """
        Initializes the DataProfiler.

        Args:
            top_n (int, optional): Number of most frequent values to track per column.
                                    Defaults to 3.
        """
        self.top_n = top_n

    def profile(self, df: pl.DataFrame | SparkDataFrame, output: str = "print") -> pl.DataFrame | SparkDataFrame | None:
        """
        Routes the DataFrame to the correct profiling engine based on its type.

        Args:
            df (pl.DataFrame | SparkDataFrame): The Polars or PySpark DataFrame to profile.
            output (str, optional): The output format. 'print' prints to console, 'dataframe'
                                    returns a summary DataFrame. Defaults to "print".

        Returns:
            pl.DataFrame | SparkDataFrame | None: A summary DataFrame if output is 'dataframe',
                                                  otherwise None.

        Raises:
            ValueError: If the input DataFrame type is not supported.
        """
        df_type = str(type(df)).lower()

        if "polars" in df_type:
            return self._profile_polars(df, output)
        elif "pyspark" in df_type:
            return self._profile_pyspark(df, output)
        else:
            raise ValueError(f"Unsupported type: {type(df)}. Please pass Polars or PySpark.")

    def _profile_polars(self, df: pl.DataFrame, output: str) -> pl.DataFrame | None:
        """
        Internal method to profile a Polars DataFrame.

        Args:
            df (pl.DataFrame): The Polars DataFrame to profile.
            output (str): The desired output format ('print' or 'dataframe').

        Returns:
            pl.DataFrame | None: A Polars DataFrame containing the profile if output='dataframe'.
        """
        total_rows = df.height
        results = []

        if output == "print":
            _logger.info("=== POLARS DATAFRAME PROFILE ===")
            _logger.info("Shape: %d rows, %d columns\n%s", total_rows, df.width, "=" * 40)

        for col_name, dtype in df.schema.items():
            # Gather metrics
            null_count = df[col_name].null_count()
            null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0
            n_unique = df[col_name].n_unique()
            is_unique = n_unique == total_rows

            counts = df[col_name].value_counts().sort("count", descending=True).head(self.top_n)
            freq_vals = [f"{row[col_name]}: {row['count']}" for row in counts.iter_rows(named=True)]
            top_vals_str = " | ".join(freq_vals)

            if output == "print":
                _logger.info("Column: %s", col_name)
                _logger.info(
                    "  Type: %s\n  Missing: %d (%.2f%%)\n  Unique: %d (All Unique: %s)\n  Top %d: %s",
                    dtype,
                    null_count,
                    null_pct,
                    n_unique,
                    is_unique,
                    self.top_n,
                    top_vals_str,
                )
                _logger.info("-" * 40)
            else:
                results.append(
                    {
                        "column": col_name,
                        "dtype": str(dtype),
                        "missing_count": null_count,
                        "missing_pct": null_pct,
                        "unique_count": n_unique,
                        "is_all_unique": is_unique,
                        "top_values": top_vals_str,
                    }
                )

        if output == "dataframe":
            return pl.DataFrame(results)

    def _profile_pyspark(self, df: SparkDataFrame, output: str) -> SparkDataFrame | None:
        """
        Internal method to profile a PySpark DataFrame.

        Args:
            df (SparkDataFrame): The PySpark DataFrame to profile.
            output (str): The desired output format ('print' or 'dataframe').

        Returns:
            SparkDataFrame | None: A PySpark DataFrame containing the profile if output='dataframe'.
        """
        total_rows = df.count()
        results = []

        if output == "print":
            _logger.info("=== PYSPARK DATAFRAME PROFILE ===")
            _logger.info("Shape: %d rows, %d columns\n%s", total_rows, len(df.columns), "=" * 40)

        agg_exprs = []
        for c in df.columns:
            agg_exprs.append(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}_nulls"))
            agg_exprs.append(F.approx_count_distinct(c).alias(f"{c}_uniques"))

        # Executes exactly ONE Spark job to gather all nulls and unique counts
        stats_row = df.agg(*agg_exprs).collect()[0]

        for col_name in df.columns:
            dtype = df.schema[col_name].dataType.typeName()

            # Fetch the pre-calculated stats from the single row
            null_count = stats_row[f"{col_name}_nulls"] or 0
            null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0
            n_unique = stats_row[f"{col_name}_uniques"]
            is_unique_approx = n_unique >= (total_rows * 0.99)  # Approximation threshold

            # This triggers 1 job per column to get the top N frequent values
            freq_df = df.groupBy(col_name).count().orderBy(F.col("count").desc()).limit(self.top_n).collect()
            freq_vals = [f"{row[col_name]}: {row['count']}" for row in freq_df]
            top_vals_str = " | ".join(freq_vals)

            if output == "print":
                _logger.info("Column: %s", col_name)
                _logger.info(
                    "  Type: %s\n  Missing: %d (%.2f%%)\n  Approx Unique: %d\n  Top %d: %s",
                    dtype,
                    null_count,
                    null_pct,
                    n_unique,
                    self.top_n,
                    top_vals_str,
                )
                _logger.info("-" * 40)
            else:
                results.append((col_name, dtype, null_count, float(null_pct), n_unique, is_unique_approx, top_vals_str))

        if output == "dataframe":
            spark = df.sparkSession
            schema = StructType(
                [
                    StructField("column", StringType(), True),
                    StructField("dtype", StringType(), True),
                    StructField("missing_count", LongType(), True),
                    StructField("missing_pct", DoubleType(), True),
                    StructField("approx_unique_count", LongType(), True),
                    StructField("is_all_unique_approx", BooleanType(), True),
                    StructField("top_values", StringType(), True),
                ]
            )
            return spark.createDataFrame(results, schema=schema)


def frame_shape(df: SparkDataFrame) -> tuple[int, int]:
    """
    Calculates and prints the shape of a PySpark DataFrame in a Polars/Pandas-like format.

    Args:
        df (SparkDataFrame): The PySpark DataFrame to measure.

    Returns:
        tuple[int, int]: A tuple containing (row_count, column_count).
    """
    rows = df.count()
    cols = len(df.columns)
    _logger.info("Shape: (%d, %d)", rows, cols)
    return (rows, cols)


def clean_column_names(df: SparkDataFrame) -> SparkDataFrame:
    """
    Renames DataFrame columns to be Delta-compatible by replacing special characters.

    Keeps only alphanumeric characters and underscores, and collapses multiple
    consecutive underscores into a single one. Ensures unique column names to
    prevent ambiguous reference errors.

    Args:
        df (SparkDataFrame): The PySpark DataFrame with potentially invalid column names.

    Returns:
        SparkDataFrame: A new PySpark DataFrame with sanitized, unique column names.
    """
    # Pattern: match anything that is NOT a-z, A-Z, 0-9, or _
    pattern = re.compile(r"[^a-zA-Z0-9_]")

    new_cols = []
    seen = set()

    for col in df.columns:
        clean_name = pattern.sub("_", col)
        # Collapse multiple underscores into one
        clean_name = re.sub(r"_+", "_", clean_name).strip("_")

        # Fallback if the name was entirely special characters
        if not clean_name:
            clean_name = "column"

        # Deduplicate collisions to avoid Spark AnalysisExceptions
        final_name = clean_name
        counter = 1
        while final_name in seen:
            final_name = f"{clean_name}_{counter}"
            counter += 1

        seen.add(final_name)
        new_cols.append(final_name)

    return df.toDF(*new_cols)


def keep_duplicates(df: SparkDataFrame, subset: list[str] | str) -> SparkDataFrame:
    """
    Filters the DataFrame to keep only rows that have duplicates based on specified columns.
    Optimized to prevent full-table shuffles by using a broadcast join.

    Args:
        df (SparkDataFrame): The PySpark DataFrame to filter.
        subset (list[str] | str): Column name(s) to check for duplicates.

    Returns:
        SparkDataFrame: A new PySpark DataFrame containing only rows with duplicate values
                        in the specified subset columns.
    """
    if isinstance(subset, str):
        subset = [subset]

    duplicate_keys = df.groupBy(*subset).count().filter(F.col("count") > 1).drop("count")

    return df.join(F.broadcast(duplicate_keys), on=subset, how="inner")


def is_unique(df: SparkDataFrame, column_name: str) -> bool:
    """
    Checks if all values in a specified column of a PySpark DataFrame are unique.

    Uses a short-circuiting groupBy operation for performance.

    Args:
        df (SparkDataFrame): The PySpark DataFrame to check.
        column_name (str): The name of the column to evaluate.

    Returns:
        bool: True if the column is entirely unique, False otherwise.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Group by the column, count occurrences, and filter for counts > 1.
    # take(1) triggers a short-circuit, making it much faster than a full distinct().count()
    # if a duplicate exists.
    duplicates = df.groupBy(column_name).count().filter(F.col("count") > 1).take(1)

    unique = len(duplicates) == 0

    if unique:
        _logger.info("Column '%s' is unique: yes", column_name)
    else:
        _logger.info("Column '%s' is unique: no", column_name)

    return unique


def glimpse(df: SparkDataFrame, n: int = 5, truncate: int = 75) -> None:
    """
    Prints a concise, vertical summary of a Spark DataFrame.

    Similar to R's dplyr::glimpse or Polars' .glimpse(). Shows the number of rows
    and columns, followed by column names, data types, and the first few values.
    Note: This triggers a Spark job to count rows and fetch the top `n` records.

    Args:
        df (SparkDataFrame): The PySpark DataFrame to glimpse.
        n (int, optional): Number of rows to preview. Defaults to 5.
        truncate (int, optional): Maximum string length for the preview values. Defaults to 75.

    Returns:
        None: This function prints output and does not return a value.
    """
    # 1. Get shape (triggers a Spark job)
    rows = df.count()
    cols = len(df.columns)

    _logger.info("Rows: %d", rows)
    _logger.info("Columns: %d", cols)

    if rows == 0 or cols == 0:
        return

    # 2. Fetch the first n rows to the driver
    head_rows = df.take(n)
    dtypes = dict(df.dtypes)

    # 3. Calculate max lengths for nice visual alignment
    max_col_len = max([len(c) for c in df.columns])
    max_type_len = max([len(t) for _, t in df.dtypes])

    # 4. Print the aligned schema and data preview
    for col_name in df.columns:
        dtype = dtypes[col_name]

        # Extract and stringify values for this column
        # Replace None with "null" for clearer visual representation
        vals = ["null" if getattr(row, col_name) is None else str(getattr(row, col_name)) for row in head_rows]
        vals_str = ", ".join(vals)

        # Truncate if the preview string gets too long
        if len(vals_str) > truncate:
            vals_str = vals_str[:truncate] + "..."

        # Formatting: $ col_name <type> values
        aligned_col = col_name.ljust(max_col_len)
        aligned_type = f"<{dtype}>".ljust(max_type_len + 2)  # +2 for the angle brackets

        _logger.info("$ %s %s %s", aligned_col, aligned_type, vals_str)


def apply_column_comments(spark: SparkSession, table_name: str, comments: dict[str, str], verbose: bool = True) -> None:
    """
    Applies column comments to a specified table only if the comment has changed.

    Logic workflow:
    1. Fetches current table schema and existing comments.
    2. Warns about table columns missing from the input dictionary.
    3. Iterates through the input dictionary, skipping updates if:
       - The input comment is None or an empty string.
       - The column doesn't exist in the table.
       - The new comment matches the existing comment exactly.
    4. Executes SQL only for actual changes.

    Args:
        spark (SparkSession): The active PySpark session.
        table_name (str): Fully-qualified target table name.
        comments (dict[str, str]): A dictionary mapping column names to their new comment strings.
        verbose (bool, optional): If True, prints details about skipped vs updated columns.
                                  Defaults to True.

    Returns:
        None: This function applies comments in-place and does not return a value.
    """
    # 1. Get existing schema and metadata
    try:
        # We access schema directly without triggering a job
        table_schema = spark.table(table_name).schema
    except Exception as e:
        # Table access failure — cannot proceed without schema, so log and return
        # silently rather than crashing the caller's workflow.
        _logger.error("Error accessing table '%s': %s", table_name, e)
        return

    # Map existing columns to their current comments (default to empty string if None)
    existing_col_map = {}
    for field in table_schema.fields:
        # Metadata 'comment' key might be missing, or might be None
        current_comment = field.metadata.get("comment", "")
        existing_col_map[field.name] = current_comment if current_comment else ""

    existing_col_names = set(existing_col_map.keys())
    input_col_names = set(comments.keys())

    # 2. Warn for existing table columns that were NOT provided in the input dict
    missing_in_input = existing_col_names.difference(input_col_names)
    if missing_in_input and verbose:
        _logger.warning("--- [Audit] Columns in '%s' missing from input dictionary: ---", table_name)
        _logger.warning("%s\n", ", ".join(sorted(missing_in_input)))

    # Track operations for summary
    updated_count = 0
    skipped_count = 0

    # 3. Process the comments
    for col, new_comment_raw in comments.items():
        # Normalize input: Treat None as empty string for comparison
        new_comment = new_comment_raw if new_comment_raw else ""

        # CONDITION A: Check if input is "empty" (per requirement)
        if new_comment == "":
            if verbose:
                _logger.info("Skipping '%s': Input comment is empty.", col)
            continue

        # CONDITION B: Check if column actually exists in table
        if col not in existing_col_names:
            _logger.warning("Column '%s' defined in comments but NOT found in table '%s'", col, table_name)
            continue

        # CONDITION C: Compare New vs Old (The optimization)
        current_comment = existing_col_map[col]

        if new_comment == current_comment:
            skipped_count += 1
            # Debug log only if you really want to see noise, otherwise keep silent or minimal
            # if verbose: print(f"Matches existing: {col}")
            continue

        # If we reach here, we need to update
        if verbose:
            _logger.info("Updating '%s': \n   Old: '%s' \n   New: '%s'", col, current_comment, new_comment)

        # Escape single quotes for SQL safety
        escaped_comment = new_comment.replace("'", "''")

        sql = f"""
        COMMENT ON COLUMN {table_name}.{col}
        IS '{escaped_comment}'
        """

        try:
            spark.sql(sql)
            updated_count += 1
        except Exception as e:
            # Individual column update failure — log and continue so remaining
            # columns still get their comments applied.
            _logger.error("Failed to update column '%s': %s", col, e)

    if verbose:
        _logger.info("\n--- Done. Updated: %d | Skipped (No Change): %d ---", updated_count, skipped_count)


def display2(df: Any, is_dev: bool | None = None) -> None:
    """
    Displays a DataFrame (Spark, Polars, or Pandas) using Databricks' built-in display()
    only if is_dev is True. If is_dev is not explicitly provided, reads IS_DEV from the
    notebook namespace. Defaults to True if IS_DEV is not set.

    Args:
        df: The DataFrame to display.
        is_dev (bool, optional): Flag indicating if the environment is development.
                                 Defaults to True if IS_DEV is not set.

    Returns:
        None: This function displays the DataFrame and does not return a value.
    """
    # 1. Resolve IS_DEV using shared logic
    is_dev = _resolve_is_dev(is_dev)

    if not is_dev:
        _logger.info("Skipping display (IS_DEV is False or not found).")
        return

    # 2. Get the true Databricks display function
    display_func = None
    try:
        # The official way to import Databricks display into an external module (DBR 11.0+)
        from dbruntime.display import display as display_func
    except ImportError:
        # Fallback to the notebook namespace if dbruntime isn't accessible
        display_func = _get_notebook_var("display")

    # 3. Handle Polars to ensure it renders correctly in the Databricks UI
    display_target = df
    if "polars" in str(type(df)).lower() and hasattr(df, "to_pandas"):
        display_target = df.to_pandas()

    # 4. Execute display with a clean fallback
    if display_func and callable(display_func):
        try:
            display_func(display_target)
            return  # Success!
        except Exception as e:
            # Databricks display() may fail outside notebook context (e.g. pytest);
            # log and fall back to local rendering rather than crashing.
            _logger.warning("Databricks display() failed: %s. Falling back...", e)

    # If we reach here, we are probably testing locally (e.g., in pytest)
    # where Databricks widgets don't exist.
    if hasattr(df, "show"):
        df.show()
    elif hasattr(df, "glimpse"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            df.glimpse()
        _logger.info(buf.getvalue().rstrip())
    else:
        _logger.info("%s", df)
