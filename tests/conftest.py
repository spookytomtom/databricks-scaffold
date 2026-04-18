import pytest
import shutil
import os
import errno
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from pyspark.sql import SparkSession
from databricks_scaffold.core import VolumeSpiller


class _FakeNotFound(Exception):
    """Stand-in for databricks.sdk.errors.NotFound used by the fake Files API."""


class FakeFilesAPI:
    """
    Emulates databricks.sdk.WorkspaceClient().files against the local filesystem.
    Every method signature matches the real SDK so the production code cannot tell
    the difference. Raises _FakeNotFound for missing paths so tests can assert on it.
    """

    def upload_from(self, file_path, source_path, overwrite=True, use_parallel=True, parallelism=None, part_size=None):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not overwrite and os.path.exists(file_path):
            raise FileExistsError(file_path)
        shutil.copyfile(source_path, file_path)

    def download_to(self, file_path, destination, use_parallel=True, parallelism=None):
        if not os.path.exists(file_path):
            raise _FakeNotFound(file_path)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        shutil.copyfile(file_path, destination)

    def create_directory(self, directory_path):
        os.makedirs(directory_path, exist_ok=True)

    def get_directory_metadata(self, directory_path):
        if not os.path.isdir(directory_path):
            raise _FakeNotFound(directory_path)
        return SimpleNamespace(path=directory_path)

    def list_directory_contents(self, directory_path):
        if not os.path.isdir(directory_path):
            raise _FakeNotFound(directory_path)
        for name in sorted(os.listdir(directory_path)):
            full = os.path.join(directory_path, name)
            yield SimpleNamespace(name=name, path=full, is_directory=os.path.isdir(full))

    def delete(self, file_path):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            raise _FakeNotFound(file_path)

    def delete_directory(self, directory_path):
        try:
            os.rmdir(directory_path)
        except FileNotFoundError:
            raise _FakeNotFound(directory_path)
        except OSError as e:
            if e.errno == errno.ENOTEMPTY:
                raise
            raise


class FakeWorkspaceClient:
    def __init__(self):
        self.files = FakeFilesAPI()


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
    
    # Initialize the class — patch spark.sql to avoid Databricks-only
    # CREATE/DROP VOLUME commands that fail in a local SparkSession
    with patch.object(spark, "sql", return_value=None):
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


@pytest.fixture
def spiller_connect(spiller, monkeypatch):
    """
    Returns a VolumeSpiller with Databricks Connect mode forced on and a fake
    WorkspaceClient attached. Volume paths are still local tmp_path dirs (inherited
    from the `spiller` fixture), so the fake Files API operates on them directly.
    """
    spiller._is_connect = True
    spiller._w = FakeWorkspaceClient()
    return spiller