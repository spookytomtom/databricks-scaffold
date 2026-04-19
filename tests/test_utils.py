# Fixtures used: spark (pyspark.sql.SparkSession), capsys (pytest built-in)
from unittest.mock import patch

from databricks_scaffold._internal import _resolve_is_dev
from databricks_scaffold.utils import clean_column_names, glimpse, keep_duplicates


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


def test_clean_column_names(spark):
    """Test clean_column_names removes spaces, dashes, and dots from column names."""
    data = [(1, "foo")]
    df = spark.createDataFrame(data, ["ID #", "First.Name-Extra"])

    clean_df = clean_column_names(df)
    assert clean_df.columns == ["ID", "First_Name_Extra"]


def test_keep_duplicates(spark):
    """Test keep_duplicates returns only rows that have duplicates."""
    data = [(1, "A"), (2, "B"), (1, "A"), (3, "C")]
    df = spark.createDataFrame(data, ["id", "val"])

    dupes = keep_duplicates(df, subset=["id", "val"])
    assert dupes.count() == 2
    assert all(row["id"] == 1 for row in dupes.collect())


def test_glimpse(spark, capsys):
    """Test glimpse prints row count, column count, and sample data."""
    data = [(1, "Alice", 100.5), (2, "Bob", None), (3, "Charlie", 300.0)]
    df = spark.createDataFrame(data, ["id", "name", "amount"])

    glimpse(df, n=2)

    captured = capsys.readouterr()
    output = captured.out

    assert "Rows: 3" in output
    assert "Columns: 3" in output
    assert "$ id" in output
    assert "<bigint>" in output
    assert "1, 2" in output
    assert "null" in output
