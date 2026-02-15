import polars as pl
import pytest
from datetime import datetime

def test_initialization(spiller):
    """Test if directories are correctly set up."""
    assert "test_vol" in spiller.volume_root
    assert spiller.is_dev is True

def test_checkpoint_roundtrip_polars(spiller):
    """Save a Polars DF to volume and load it back."""
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "val": ["a", "b", "c"]
    })
    
    spiller.save_checkpoint_pl(df, name="test_pl_checkpoint")
    
    # Verify file exists on "volume" (our temp dir)
    files = spiller.list_checkpoints(storage="volume")
    assert "test_pl_checkpoint" in files
    
    # Load back
    loaded_df = spiller.load_checkpoint_pl("test_pl_checkpoint", eager=True)
    assert df.equals(loaded_df)

def test_nanosecond_autofix(spiller, capsys):
    """Ensure ns timestamps are auto-converted to ms (Spark compatibility)."""
    # create a DF with nanosecond precision
    df = pl.DataFrame({
        "ts": [datetime(2023, 1, 1, 12, 0, 0)]
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
    
    # This should trigger the auto-fix print and conversion
    spiller.save_checkpoint_pl(df, name="ns_test")
    
    # Check if the warning was printed
    captured = capsys.readouterr()
    assert "Auto-fix: Casting" in captured.out
    
    # Load back and verify it's now ms or us (parquet standard), not ns
    loaded = spiller.load_checkpoint_pl("ns_test")
    dtype = loaded.schema["ts"]
    assert dtype != pl.Datetime("ns")

def test_spark_to_polars_conversion(spiller, spark):
    """Test moving data from Spark to Polars."""
    spark_df = spark.createDataFrame([(1, "foo"), (2, "bar")], ["id", "txt"])
    
    # Execute conversion
    pl_df = spiller.spark_to_polars(spark_df, eager=True, cleanup=True)
    
    assert isinstance(pl_df, pl.DataFrame)
    assert pl_df.shape == (2, 2)
    assert pl_df.filter(pl.col("id") == 1)["txt"][0] == "foo"

def test_polars_to_spark_conversion(spiller):
    """Test moving data from Polars to Spark."""
    pl_df = pl.DataFrame({
        "id": [10, 20],
        "val": [1.1, 2.2]
    })
    
    spark_df = spiller.polars_to_spark(pl_df, cleanup=True)
    
    # Collect to local list to verify
    rows = spark_df.sort("id").collect()
    assert rows[0]["id"] == 10
    assert rows[1]["val"] == 2.2

def test_error_handling_invalid_storage(spiller):
    """Test that invalid storage options raise errors."""
    df = pl.DataFrame({"a": [1]})
    
    with pytest.raises(ValueError, match="storage must be"):
        spiller.save_checkpoint_pl(df, "bad_store", storage="cloud")