import re
from pyspark.sql import DataFrame as SparkDataFrame, functions as F, Window

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