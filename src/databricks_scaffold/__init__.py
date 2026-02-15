from typing import Union

# Import the logic
from .core import VolumeSpiller
from .utils import (
    frame_shape,
    clean_column_names,
    keep_duplicates
)

# Import types for aliasing
from pyspark.sql import DataFrame as SparkDataFrame
import polars as pl

# Create convenient aliases for your users
# This helps when they want to type-hint their own functions using your library
PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]

__all__ = [
    "VolumeSpiller",
    "frame_shape",
    "clean_column_names",
    "keep_duplicates",
    "SparkDataFrame",
    "PolarsFrame"
]