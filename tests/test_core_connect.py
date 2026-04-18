import pytest
from databricks_scaffold.core import _is_databricks_connect


def test_local_sparksession_is_not_connect(spark):
    """A plain local SparkSession must return False."""
    assert _is_databricks_connect(spark) is False


def test_returns_false_when_sdk_not_installed(spark, monkeypatch):
    """Missing databricks.connect.session import must return False, not raise."""
    import sys
    monkeypatch.setitem(sys.modules, "databricks.connect.session", None)
    assert _is_databricks_connect(spark) is False


def test_spiller_caches_is_connect_false(spiller):
    assert spiller._is_connect is False


def test_spiller_workspace_client_is_lazy(spiller_connect):
    """
    The fake WorkspaceClient injected by the fixture should be the one returned,
    proving that _workspace is an attribute read (not a hard-coded construction).
    """
    assert spiller_connect._workspace is spiller_connect._w


import os


def test_volume_mkdirs_uses_files_api_under_connect(spiller_connect):
    target = f"{spiller_connect.volume_root}/new_dir/nested"
    spiller_connect._volume_mkdirs(target)
    # Our fake Files API delegates to os.makedirs, so the dir should exist on disk
    assert os.path.isdir(target)


def test_volume_mkdirs_uses_os_on_cluster(spiller, tmp_path):
    target = f"{spiller.volume_root}/on_cluster_dir"
    spiller._volume_mkdirs(target)
    assert os.path.isdir(target)


def test_volume_exists_true_when_dir_present(spiller_connect):
    target = f"{spiller_connect.volume_root}/present"
    os.makedirs(target)
    assert spiller_connect._volume_exists(target) is True


def test_volume_exists_false_when_dir_missing(spiller_connect):
    assert spiller_connect._volume_exists(f"{spiller_connect.volume_root}/absent") is False


def test_volume_listdir_returns_entry_names(spiller_connect):
    root = spiller_connect.volume_root
    os.makedirs(f"{root}/ckpt_a")
    os.makedirs(f"{root}/ckpt_b")
    names = spiller_connect._volume_listdir(root)
    assert sorted(names) == ["ckpt_a", "ckpt_b"]


def test_volume_listdir_returns_empty_for_missing_dir(spiller_connect):
    assert spiller_connect._volume_listdir(f"{spiller_connect.volume_root}/absent") == []


def test_volume_rmtree_removes_nested_files_and_dir(spiller_connect):
    target = f"{spiller_connect.volume_root}/to_delete"
    os.makedirs(target)
    with open(f"{target}/a.parquet", "wb") as f:
        f.write(b"x")
    with open(f"{target}/b.parquet", "wb") as f:
        f.write(b"y")
    spiller_connect._volume_rmtree(target)
    assert not os.path.exists(target)


def test_volume_rmtree_is_silent_when_missing(spiller_connect):
    spiller_connect._volume_rmtree(f"{spiller_connect.volume_root}/never_existed")
