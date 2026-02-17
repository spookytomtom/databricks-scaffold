import re
from pyspark.sql import DataFrame as SparkDataFrame, functions as F, Window, SparkSession
from typing import Dict 

def frame_shape(df: SparkDataFrame) -> tuple[int, int]:
    """
    Returns the shape (rows, columns) of a Spark DataFrame.
    Prints the result in a Polars-like format for quick debugging.
    """
    rows = df.count()
    cols = len(df.columns)
    print(f"Shape: ({rows}, {cols})")
    return (rows, cols)

def clean_column_names(df: SparkDataFrame) -> SparkDataFrame:
    """
    Renames columns to be Delta-compatible. 
    Keeps only alphanumeric characters and underscores.
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
    Filters the DataFrame to keep only rows that have duplicates 
    based on the specified columns.
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

def apply_column_comments(
    spark: SparkSession,
    table_name: str,
    comments: Dict[str, str],
    verbose: bool = True
) -> None:
    """
    Apply column comments on a table, but ONLY if the comment has changed.
    
    Logic:
    1. Fetches current table schema and existing comments.
    2. Warns about table columns missing from your input dictionary.
    3. Iterates through your dictionary:
       - Skips if the input comment is None or empty string "".
       - Skips if the column doesn't exist in the table.
       - Skips if the new comment matches the existing comment exactly.
    4. Executes SQL only for actual changes.

    Parameters
    ----------
    spark : SparkSession
    table_name : str
        Fully-qualified table name.
    comments : dict
        Mapping {column_name: comment_string}.
    verbose : bool
        If True, prints details about skipped vs updated columns.
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