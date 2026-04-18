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


def test_upload_dir_to_volume_copies_all_parquet_files(spiller_connect, tmp_path):
    local_src = tmp_path / "src"
    local_src.mkdir()
    (local_src / "part-0.parquet").write_bytes(b"A")
    (local_src / "part-1.parquet").write_bytes(b"B")
    (local_src / "_SUCCESS").write_bytes(b"")  # non-parquet; must be ignored

    remote = f"{spiller_connect.volume_root}/uploaded"
    spiller_connect._upload_dir_to_volume(str(local_src), remote)

    assert os.path.exists(f"{remote}/part-0.parquet")
    assert os.path.exists(f"{remote}/part-1.parquet")
    assert not os.path.exists(f"{remote}/_SUCCESS")


def test_download_volume_dir_copies_all_parquet_files(spiller_connect, tmp_path):
    remote = f"{spiller_connect.volume_root}/remote"
    os.makedirs(remote)
    with open(f"{remote}/part-0.parquet", "wb") as f:
        f.write(b"A")
    with open(f"{remote}/part-1.parquet", "wb") as f:
        f.write(b"B")
    with open(f"{remote}/_SUCCESS", "wb") as f:
        f.write(b"")  # non-parquet; must be ignored

    local_dst = tmp_path / "dst"
    local_dst.mkdir()
    spiller_connect._download_volume_dir(remote, str(local_dst))

    assert (local_dst / "part-0.parquet").exists()
    assert (local_dst / "part-1.parquet").exists()
    assert not (local_dst / "_SUCCESS").exists()


def test_teardown_cleans_both_local_and_volume_tracked_dirs(spiller_connect, tmp_path):
    local_dir = tmp_path / "staging"
    local_dir.mkdir()
    (local_dir / "x.parquet").write_bytes(b"x")

    volume_dir = f"{spiller_connect.volume_root}/spill_track"
    os.makedirs(volume_dir)
    with open(f"{volume_dir}/y.parquet", "wb") as f:
        f.write(b"y")

    spiller_connect._active_local_dirs.append(str(local_dir))
    spiller_connect._active_volume_dirs.append(volume_dir)

    spiller_connect.teardown()

    assert not local_dir.exists()
    assert not os.path.exists(volume_dir)


import polars as pl


def test_spark_to_polars_connect_eager_returns_dataframe(spiller_connect, spark):
    spark_df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "txt"])
    pl_df = spiller_connect.spark_to_polars(spark_df, eager=True, cleanup=True)
    assert isinstance(pl_df, pl.DataFrame)
    assert pl_df.shape == (2, 2)
    assert sorted(pl_df["id"].to_list()) == [1, 2]


def test_spark_to_polars_connect_cleanup_true_removes_local_staging(spiller_connect, spark, tmp_path):
    spark_df = spark.createDataFrame([(1,)], ["id"])
    _ = spiller_connect.spark_to_polars(spark_df, eager=True, cleanup=True)
    assert spiller_connect._active_local_dirs == []
    leftover = list(spiller_connect.local_base_dir.rglob("*.parquet"))
    assert leftover == []


def test_spark_to_polars_connect_lazy_tracks_staging_dir(spiller_connect, spark):
    spark_df = spark.createDataFrame([(1,)], ["id"])
    lf = spiller_connect.spark_to_polars(spark_df, eager=False)
    assert isinstance(lf, pl.LazyFrame)
    assert len(spiller_connect._active_local_dirs) == 1
    staging = spiller_connect._active_local_dirs[0]
    assert any(name.endswith(".parquet") for name in os.listdir(staging))
    df = lf.collect()
    assert df.shape == (1, 1)
