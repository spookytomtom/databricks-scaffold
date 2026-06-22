# Fixtures used: spark (pyspark.sql.SparkSession), caplog (pytest built-in)
from unittest.mock import patch

import polars as pl
import pytest

from databricks_scaffold._internal import _resolve_is_dev
from databricks_scaffold.utils import (
    DataProfiler,
    apply_column_comments,
    clean_column_names,
    frame_shape,
    glimpse,
    is_unique,
    keep_duplicates,
)


def test_resolve_is_dev_explicit():
    """Test _resolve_is_dev returns explicit boolean values unchanged."""
    assert _resolve_is_dev(True) is True
    assert _resolve_is_dev(False) is False


def test_resolve_is_dev_default():
    """Test _resolve_is_dev defaults to True when no notebook var is set."""
    assert _resolve_is_dev(None) is True


def test_resolve_is_dev_string_booleans():
    """Test _resolve_is_dev parses string boolean representations correctly."""
    for falsy in ("false", "False", "FALSE", "0", "no", "f", ""):
        with patch("databricks_scaffold._internal._get_notebook_var", return_value=falsy):
            assert _resolve_is_dev(None) is False, f"Expected False for {falsy!r}"
    for truthy in ("true", "True", "1", "yes"):
        with patch("databricks_scaffold._internal._get_notebook_var", return_value=truthy):
            assert _resolve_is_dev(None) is True, f"Expected True for {truthy!r}"


@pytest.mark.requires_pyspark
def test_clean_column_names(spark):
    """Test clean_column_names removes spaces, dashes, and dots from column names."""
    data = [(1, "foo")]
    df = spark.createDataFrame(data, ["ID #", "First.Name-Extra"])

    clean_df = clean_column_names(df)
    assert clean_df.columns == ["ID", "First_Name_Extra"]


@pytest.mark.requires_pyspark
def test_keep_duplicates(spark):
    """Test keep_duplicates returns only rows that have duplicates."""
    data = [(1, "A"), (2, "B"), (1, "A"), (3, "C")]
    df = spark.createDataFrame(data, ["id", "val"])

    dupes = keep_duplicates(df, subset=["id", "val"])
    assert dupes.count() == 2
    assert all(row["id"] == 1 for row in dupes.collect())


@pytest.mark.requires_pyspark
def test_glimpse(spark, caplog):
    """Test glimpse prints row count, column count, and sample data."""
    caplog.set_level("INFO")
    data = [(1, "Alice", 100.5), (2, "Bob", None), (3, "Charlie", 300.0)]
    df = spark.createDataFrame(data, ["id", "name", "amount"])

    glimpse(df, n=2)

    assert "Rows: 3" in caplog.text
    assert "Columns: 3" in caplog.text
    assert "$ id" in caplog.text
    assert "<bigint>" in caplog.text
    assert "1, 2" in caplog.text
    assert "null" in caplog.text


def test_data_profiler_polars_prints_summary(caplog):
    caplog.set_level("INFO")
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "region": ["EMEA", "AMER", "EMEA", "APAC", "AMER"],
    })
    profiler = DataProfiler(top_n=2)
    result = profiler.profile(df, output="print")
    assert result is None
    assert "POLARS DATAFRAME PROFILE" in caplog.text
    assert "Shape: 5 rows, 2 columns" in caplog.text
    assert "EMEA" in caplog.text


def test_data_profiler_polars_returns_dataframe():
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "region": ["EMEA", "AMER", "EMEA", "APAC", "AMER"],
    })
    profiler = DataProfiler(top_n=2)
    result = profiler.profile(df, output="dataframe")
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (2, 7)
    assert set(result["column"]) == {"id", "region"}
    assert result.filter(pl.col("column") == "id")["is_all_unique"][0] is True


@pytest.mark.requires_pyspark
def test_frame_shape(spark):
    df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
    shape = frame_shape(df)
    assert shape == (2, 2)


@pytest.mark.requires_pyspark
def test_is_unique_true(spark):
    df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
    assert is_unique(df, "id") is True


@pytest.mark.requires_pyspark
def test_is_unique_false(spark):
    df = spark.createDataFrame([(1,), (2,), (1,)], ["id"])
    assert is_unique(df, "id") is False


@pytest.mark.requires_pyspark
def test_is_unique_missing_column(spark):
    df = spark.createDataFrame([(1,)], ["id"])
    with pytest.raises(ValueError, match="Column 'missing' not found"):
        is_unique(df, "missing")


@pytest.mark.requires_pyspark
def test_keep_duplicates_string_subset(spark):
    df = spark.createDataFrame(
        [(1, "A"), (2, "B"), (1, "A"), (3, "C")],
        ["id", "val"],
    )
    dupes = keep_duplicates(df, subset="id")
    assert dupes.count() == 2


@pytest.mark.requires_pyspark
def test_clean_column_names_collision(spark):
    df = spark.createDataFrame(
        [(1, 2)],
        ["ID #", "ID%"],
    )
    clean_df = clean_column_names(df)
    assert clean_df.columns == ["ID", "ID_1"]


@pytest.mark.requires_pyspark
def test_apply_column_comments(spark, caplog):
    caplog.set_level("INFO")
    spark.sql("CREATE OR REPLACE TEMP VIEW test_comments AS SELECT 1 AS id, 'x' AS name")
    comments = {"id": "Unique identifier", "name": ""}
    apply_column_comments(spark, "test_comments", comments, verbose=True)
    assert "Updating 'id'" in caplog.text
    assert "Skipping 'name': Input comment is empty" in caplog.text
