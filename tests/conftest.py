import errno
import importlib.util
import os
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import databricks_scaffold.core as _core
from databricks_scaffold.core import VolumeSpiller

try:
    from databricks.sdk.errors import NotFound as _SdkNotFound
except ImportError:

    class _SdkNotFound(OSError):  # type: ignore[assignment]
        pass


class FakeFilesAPI:
    """
    Emulates databricks.sdk.WorkspaceClient().files against the local filesystem.
    Every method signature matches the real SDK so the production code cannot tell
    the difference. Raises the real databricks.sdk.errors.NotFound (an OSError
    subclass) for missing paths, matching what the SDK raises in production.
    """

    def upload_from(self, file_path, source_path, overwrite=True, use_parallel=True, parallelism=None, part_size=None):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not overwrite and os.path.exists(file_path):
            raise FileExistsError(file_path)
        shutil.copyfile(source_path, file_path)

    def download_to(self, file_path, destination, use_parallel=True, parallelism=None):
        if not os.path.exists(file_path):
            raise _SdkNotFound(file_path)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        shutil.copyfile(file_path, destination)

    def create_directory(self, directory_path):
        os.makedirs(directory_path, exist_ok=True)

    def get_directory_metadata(self, directory_path):
        if not os.path.isdir(directory_path):
            raise _SdkNotFound(directory_path)
        return SimpleNamespace(path=directory_path)

    def list_directory_contents(self, directory_path):
        if not os.path.isdir(directory_path):
            raise _SdkNotFound(directory_path)
        for name in sorted(os.listdir(directory_path)):
            full = f"{directory_path}/{name}"
            yield SimpleNamespace(name=name, path=full, is_directory=os.path.isdir(full))

    def delete(self, file_path):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            raise _SdkNotFound(file_path)

    def delete_directory(self, directory_path):
        try:
            os.rmdir(directory_path)
        except FileNotFoundError:
            raise _SdkNotFound(directory_path)
        except OSError as e:
            if e.errno == errno.ENOTEMPTY:
                raise
            raise


class FakeWorkspaceClient:
    def __init__(self):
        self.files = FakeFilesAPI()


def _local_spark_available():
    """True only when standalone pyspark (not databricks-connect) is present."""
    try:
        if importlib.util.find_spec("databricks.connect") is not None:
            return False
    except ModuleNotFoundError:
        pass
    return importlib.util.find_spec("pyspark") is not None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_pyspark: test needs a real local SparkSession (standalone pyspark, not databricks-connect)",
    )


def pytest_collection_modifyitems(config, items):
    if not _local_spark_available():
        skip = pytest.mark.skip(reason="requires standalone pyspark — skipped in databricks-connect environments")
        for item in items:
            if item.get_closest_marker("requires_pyspark"):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def spark():
    """MagicMock stand-in for SparkSession. Tests needing a real session are marked requires_pyspark."""
    return MagicMock()


@pytest.fixture
def spiller(spark, tmp_path):
    """Creates a VolumeSpiller with a mock SparkSession and local tmp paths."""
    fake_volume_root = tmp_path / "Volumes" / "main" / "default" / "test_vol"
    fake_volume_root.mkdir(parents=True, exist_ok=True)

    # MagicMock handles spark.sql() automatically — no patch needed
    spiller_instance = VolumeSpiller(
        spark=spark,
        catalog="main",
        schema="default",
        volume_name="test_vol",
        is_dev=True,
    )

    spiller_instance.volume_root = str(fake_volume_root)
    spiller_instance.local_base_dir = tmp_path / "local_spill"
    spiller_instance.local_base_dir.mkdir(parents=True, exist_ok=True)

    yield spiller_instance

    spiller_instance.teardown()


@pytest.fixture
def spiller_connect(spark, tmp_path, monkeypatch):
    """VolumeSpiller with Connect mode active from construction, backed by FakeWorkspaceClient.

    Patches _is_databricks_connect before __init__ runs so any Connect-specific
    branching inside the constructor is exercised, not bypassed.
    """
    monkeypatch.setattr(_core, "_is_databricks_connect", lambda _: True)

    fake_volume_root = tmp_path / "Volumes" / "main" / "default" / "test_vol"
    fake_volume_root.mkdir(parents=True, exist_ok=True)

    spiller_instance = VolumeSpiller(
        spark=spark,
        catalog="main",
        schema="default",
        volume_name="test_vol",
        is_dev=True,
        workspace_client=FakeWorkspaceClient(),
    )

    spiller_instance.volume_root = str(fake_volume_root)
    spiller_instance.local_base_dir = tmp_path / "local_spill"
    spiller_instance.local_base_dir.mkdir(parents=True, exist_ok=True)

    yield spiller_instance

    spiller_instance.teardown()
