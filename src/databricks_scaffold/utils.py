import re
from pyspark.sql import DataFrame as SparkDataFrame, functions as F, Window, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, BooleanType
from typing import Dict 
import polars as pl

class DataProfiler:
    """
    A utility class for generating summary profiles of Polars and PySpark DataFrames, 
    including missing values, unique counts, and top frequent values.
    """
    def __init__(self, top_n_freq=3):
        """
        Initializes the DataProfiler.

        Args:
            top_n_freq (int, optional): Number of most frequent values to track per column. 
                                        Defaults to 3.
        """
        self.top_n = top_n_freq

    def profile(self, df, output="print") -> pl.DataFrame | SparkDataFrame | None:
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

    def _profile_polars(self, df, output) -> pl.DataFrame | None:
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
            print(f"=== POLARS DATAFRAME PROFILE ===")
            print(f"Shape: {total_rows} rows, {df.width} columns\n" + "=" * 40)

        for col_name, dtype in df.schema.items():
            # Gather metrics
            null_count = df[col_name].null_count()
            null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0
            n_unique = df[col_name].n_unique()
            is_unique = (n_unique == total_rows)
            
            counts = df[col_name].value_counts().sort("count", descending=True).head(self.top_n)
            freq_vals = [f"{row[col_name]}: {row['count']}" for row in counts.iter_rows(named=True)]
            top_vals_str = " | ".join(freq_vals)

            if output == "print":
                print(f"Column: {col_name}")
                print(f"  Type: {dtype}\n  Missing: {null_count} ({null_pct}%)\n  Unique: {n_unique} (All Unique: {is_unique})\n  Top {self.top_n}: {top_vals_str}")
                print("-" * 40)
            else:
                results.append({
                    "column": col_name,
                    "dtype": str(dtype),
                    "missing_count": null_count,
                    "missing_pct": null_pct,
                    "unique_count": n_unique,
                    "is_all_unique": is_unique,
                    "top_values": top_vals_str
                })

        if output == "dataframe":
            return pl.DataFrame(results)

    def _profile_pyspark(self, df, output) -> SparkDataFrame | None:
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
            print(f"=== PYSPARK DATAFRAME PROFILE ===")
            print(f"Shape: {total_rows} rows, {len(df.columns)} columns\n" + "=" * 40)

        for col_name in df.columns:
            dtype = df.schema[col_name].dataType.typeName()
            
            # Gather metrics using approx_count_distinct for performance
            stats = df.agg(
                F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias("nulls"),
                F.approx_count_distinct(col_name).alias("approx_uniques")
            ).collect()[0]

            null_count = stats["nulls"] or 0
            null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0
            n_unique = stats["approx_uniques"]
            is_unique_approx = n_unique >= (total_rows * 0.99) # Approximation threshold

            freq_df = df.groupBy(col_name).count().orderBy(F.col("count").desc()).limit(self.top_n).collect()
            freq_vals = [f"{row[col_name]}: {row['count']}" for row in freq_df]
            top_vals_str = " | ".join(freq_vals)

            if output == "print":
                print(f"Column: {col_name}")
                print(f"  Type: {dtype}\n  Missing: {null_count} ({null_pct}%)\n  Approx Unique: {n_unique}\n  Top {self.top_n}: {top_vals_str}")
                print("-" * 40)
            else:
                results.append((
                    col_name, dtype, null_count, float(null_pct), 
                    n_unique, is_unique_approx, top_vals_str
                ))

        if output == "dataframe":
            spark = df.sparkSession
            schema = StructType([
                StructField("column", StringType(), True),
                StructField("dtype", StringType(), True),
                StructField("missing_count", LongType(), True),
                StructField("missing_pct", DoubleType(), True),
                StructField("approx_unique_count", LongType(), True),
                StructField("is_all_unique_approx", BooleanType(), True),
                StructField("top_values", StringType(), True)
            ])
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
    print(f"Shape: ({rows}, {cols})")
    return (rows, cols)

def clean_column_names(df: SparkDataFrame) -> SparkDataFrame:
    """
    Renames DataFrame columns to be Delta-compatible by replacing special characters.
    
    Keeps only alphanumeric characters and underscores, and collapses multiple 
    consecutive underscores into a single one.

    Args:
        df (SparkDataFrame): The PySpark DataFrame with potentially invalid column names.

    Returns:
        SparkDataFrame: A new PySpark DataFrame with sanitized column names.
    """
    # Pattern: match anything that is NOT a-z, A-Z, 0-9, or _
    pattern = re.compile(r'[^a-zA-Z0-9_]')
    
    new_cols = []
    for col in df.columns:
        clean_name = pattern.sub('_', col)
        # Optional: Collapse multiple underscores into one
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        new_cols.append(clean_name)
    
    return df.toDF(*new_cols)

def keep_duplicates(df: SparkDataFrame, subset: list[str] | str) -> SparkDataFrame:
    """
    Filters the DataFrame to keep only rows that have duplicates based on specified columns.

    Utilizes a Window function to count occurrences efficiently without performing a heavy join.

    Args:
        df (SparkDataFrame): The PySpark DataFrame to filter.
        subset (list[str] | str): The column name(s) to evaluate for duplicates.

    Returns:
        SparkDataFrame: A PySpark DataFrame containing only the duplicated rows.
    """
    if isinstance(subset, str):
        subset = [subset]
        
    # Use a Window to count occurrences without performing a heavy Join
    window_spec = Window.partitionBy(*subset)
    
    return (
        df.withColumn("_dupe_count", F.count("*").over(window_spec))
        .filter(F.col("_dupe_count") > 1)
        .drop("_dupe_count")
    )

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
    """
    # 1. Get shape (triggers a Spark job)
    rows = df.count()
    cols = len(df.columns)
    
    print(f"Rows: {rows}")
    print(f"Columns: {cols}")
    
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
        vals = [
            "null" if getattr(row, col_name) is None else str(getattr(row, col_name)) 
            for row in head_rows
        ]
        vals_str = ", ".join(vals)
        
        # Truncate if the preview string gets too long
        if len(vals_str) > truncate:
            vals_str = vals_str[:truncate] + "..."
            
        # Formatting: $ col_name <type> values
        aligned_col = col_name.ljust(max_col_len)
        aligned_type = f"<{dtype}>".ljust(max_type_len + 2) # +2 for the angle brackets
        
        print(f"$ {aligned_col} {aligned_type} {vals_str}")

def apply_column_comments(
    spark: SparkSession,
    table_name: str,
    comments: Dict[str, str],
    verbose: bool = True
) -> None:
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
        comments (Dict[str, str]): A dictionary mapping column names to their new comment strings.
        verbose (bool, optional): If True, prints details about skipped vs updated columns. 
                                  Defaults to True.
    """
    # 1. Get existing schema and metadata
    try:
        # We access schema directly without triggering a job
        table_schema = spark.table(table_name).schema
    except Exception as e:
        print(f"Error accessing table '{table_name}': {e}")
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
        print(f"--- [Audit] Columns in '{table_name}' missing from input dictionary: ---")
        print(f"{', '.join(sorted(missing_in_input))}\n")

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
                print(f"Skipping '{col}': Input comment is empty.")
            continue

        # CONDITION B: Check if column actually exists in table
        if col not in existing_col_names:
            print(f"!! Warning: Column '{col}' defined in comments but NOT found in table '{table_name}'")
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
            print(f"Updating '{col}': \n   Old: '{current_comment}' \n   New: '{new_comment}'")

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
            print(f"Failed to update column '{col}': {str(e)}")

    if verbose:
        print(f"\n--- Done. Updated: {updated_count} | Skipped (No Change): {skipped_count} ---")