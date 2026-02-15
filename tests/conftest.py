import pytest
import shutil
import os
from pathlib import Path
from pyspark.sql import SparkSession
from databricks_scaffold.core import VolumeSpiller

@pytest.fixture(scope="session")
def spark():
    """Creates a local Spark session for testing."""
    spark = (SparkSession.builder
             .master("local[1]")
             .appName("databricks-scaffold-tests")
             .config("spark.sql.shuffle.partitions", "1")
             .config("spark.default.parallelism", "1")
             .config("spark.driver.bindAddress", "127.0.0.1")
             .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture
def spiller(spark, tmp_path):
    """
    Creates a VolumeSpiller instance but patches the volume_root 
    to point to a local temporary directory.
    """
    # Create a fake volume structure inside the pytest temp dir
    fake_volume_root = tmp_path / "Volumes" / "main" / "default" / "test_vol"
    fake_volume_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize the class
    spiller_instance = VolumeSpiller(
        spark=spark,
        catalog="main",
        schema="default",
        volume_name="test_vol",
        is_dev=True
    )

    # MONKEY PATCH: Override the hardcoded /Volumes path to our temp path
    spiller_instance.volume_root = str(fake_volume_root)
    
    # Also override local_base_dir to avoid cluttering your actual /tmp
    spiller_instance.local_base_dir = tmp_path / "local_spill"
    spiller_instance.local_base_dir.mkdir(parents=True, exist_ok=True)

    yield spiller_instance
    
    # Teardown handles cleanup
    spiller_instance.teardown()