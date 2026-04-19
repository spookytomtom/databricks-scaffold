# Fixtures used: spiller (VolumeSpiller with local temp dir, _is_connect=False),
# spiller_connect (VolumeSpiller with mocked WorkspaceClient, _is_connect=True),
# spark (pyspark.sql.SparkSession), monkeypatch (pytest built-in), tmp_path (pytest built-in)
import os
from unittest.mock import MagicMock

import polars as pl
import pytest

from databricks_scaffold import core as _core
from databricks_scaffold.core import _is_databricks_connect, _retry_op


def test_local_sparksession_is_not_connect(spark):
    """A plain local SparkSession must return False."""
    assert _is_databricks_connect(spark) is False


def test_returns_false_when_sdk_not_installed(spark, monkeypatch):
    """Missing databricks.connect.session import must return False, not raise."""
    import sys

    monkeypatch.setitem(sys.modules, "databricks.connect.session", None)
    assert _is_databricks_connect(spark) is False


def test_detects_connect_session_by_module():
    """
    DatabricksSession.builder.getOrCreate() returns pyspark.sql.connect.session.SparkSession,
    not a DatabricksSession instance. Detection must identify it by module name.
    """
    ConnectSession = type("SparkSession", (object,), {})
    ConnectSession.__module__ = "pyspark.sql.connect.session"
    assert _is_databricks_connect(ConnectSession()) is True


def test_local_session_by_module_returns_false():
    """
    A pyspark.sql.session.SparkSession (on-cluster/local) must not be
    detected as Connect even though it has a different module path.
    """
    LocalSession = type("SparkSession", (object,), {})
    LocalSession.__module__ = "pyspark.sql.session"
    assert _is_databricks_connect(LocalSession()) is False


def test_spiller_caches_is_connect_false(spiller):
    assert spiller._is_connect is False


def test_spiller_workspace_client_is_lazy(spiller_connect):
    """
    The fake WorkspaceClient injected by the fixture should be the one returned,
    proving that _workspace is an attribute read (not a hard-coded construction).
    """
    assert spiller_connect._workspace is spiller_connect._workspace_client



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



@pytest.mark.requires_pyspark
def test_spark_to_polars_connect_eager_returns_dataframe(spiller_connect, spark):
    spark_df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "txt"])
    pl_df = spiller_connect.spark_to_polars(spark_df, eager=True, cleanup=True)
    assert isinstance(pl_df, pl.DataFrame)
    assert pl_df.shape == (2, 2)
    assert sorted(pl_df["id"].to_list()) == [1, 2]


@pytest.mark.requires_pyspark
def test_spark_to_polars_connect_cleanup_true_removes_local_staging(spiller_connect, spark, tmp_path):
    spark_df = spark.createDataFrame([(1,)], ["id"])
    _ = spiller_connect.spark_to_polars(spark_df, eager=True, cleanup=True)
    assert spiller_connect._active_local_dirs == []
    leftover = list(spiller_connect.local_base_dir.rglob("*.parquet"))
    assert leftover == []


@pytest.mark.requires_pyspark
def test_spark_to_polars_connect_lazy_tracks_staging_dir(spiller_connect, spark):
    spark_df = spark.createDataFrame([(1,)], ["id"])
    lf = spiller_connect.spark_to_polars(spark_df, eager=False)
    assert isinstance(lf, pl.LazyFrame)
    assert len(spiller_connect._active_local_dirs) == 1
    staging = spiller_connect._active_local_dirs[0]
    assert any(name.endswith(".parquet") for name in os.listdir(staging))
    df = lf.collect()
    assert df.shape == (1, 1)


@pytest.mark.requires_pyspark
def test_polars_to_spark_connect_roundtrip(spiller_connect):
    pl_df = pl.DataFrame({"id": [10, 20, 30], "val": [1.1, 2.2, 3.3]})
    spark_df = spiller_connect.polars_to_spark(pl_df)
    rows = spark_df.sort("id").collect()
    assert [r["id"] for r in rows] == [10, 20, 30]
    assert [r["val"] for r in rows] == [1.1, 2.2, 3.3]


def test_polars_to_spark_connect_tracks_volume_dir_for_teardown(spiller_connect):
    pl_df = pl.DataFrame({"id": [1]})
    _ = spiller_connect.polars_to_spark(pl_df)
    assert len(spiller_connect._active_volume_dirs) == 1
    vol_dir = spiller_connect._active_volume_dirs[0]
    assert os.path.exists(f"{vol_dir}/part-0.parquet")


@pytest.mark.requires_pyspark
def test_polars_to_spark_full_roundtrip_with_spark_to_polars(spiller_connect, spark):
    spark_df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "txt"])
    pl_df = spiller_connect.spark_to_polars(spark_df, eager=True, cleanup=True)
    pl_mod = pl_df.with_columns(pl.col("id") * 10)
    spark_back = spiller_connect.polars_to_spark(pl_mod)
    rows = spark_back.sort("id").collect()
    assert [r["id"] for r in rows] == [10, 20]
    assert [r["txt"] for r in rows] == ["a", "b"]


def test_resolve_path_volume_under_connect_does_not_call_os_makedirs(spiller_connect, monkeypatch):
    calls = {"os_makedirs": 0}
    real_makedirs = os.makedirs

    def tracking_makedirs(path, *a, **kw):
        if str(path).startswith(str(spiller_connect.volume_root)):
            calls["os_makedirs"] += 1
        return real_makedirs(path, *a, **kw)

    monkeypatch.setattr(os, "makedirs", tracking_makedirs)

    path, storage = spiller_connect._resolve_path("some_ckpt", "volume")
    assert storage == "volume"
    assert calls["os_makedirs"] == 0
    assert os.path.isdir(path)


def test_resolve_path_local_still_uses_os_makedirs(spiller_connect):
    path, storage = spiller_connect._resolve_path("local_ckpt", "local")
    assert storage == "local"
    assert os.path.isdir(path)


def test_save_checkpoint_pl_volume_under_connect(spiller_connect):
    df = pl.DataFrame({"id": [1, 2, 3], "val": ["x", "y", "z"]})
    spiller_connect.save_checkpoint_pl(df, name="gold_vol", storage="volume")
    assert os.path.exists(f"{spiller_connect.volume_root}/gold_vol/data.parquet")


def test_save_checkpoint_pl_volume_overwrites_stale_files(spiller_connect):
    ckpt_dir = f"{spiller_connect.volume_root}/gold_vol"
    os.makedirs(ckpt_dir)
    with open(f"{ckpt_dir}/stale.parquet", "wb") as f:
        f.write(b"stale")

    df = pl.DataFrame({"id": [1]})
    spiller_connect.save_checkpoint_pl(df, name="gold_vol", storage="volume")

    assert not os.path.exists(f"{ckpt_dir}/stale.parquet")
    assert os.path.exists(f"{ckpt_dir}/data.parquet")


def test_save_checkpoint_pl_upload_failure_preserves_old_checkpoint(spiller_connect, monkeypatch):
    """Regression for C4: if upload fails, old checkpoint must survive intact (write-then-swap)."""
    df_old = pl.DataFrame({"id": [1, 2, 3]})
    df_new = pl.DataFrame({"id": [10, 20, 30]})

    spiller_connect.save_checkpoint_pl(df_old, name="ckpt_c4", storage="volume")

    def _fail(*args, **kwargs):
        raise RuntimeError("simulated network error")

    monkeypatch.setattr(spiller_connect._workspace_client.files, "upload_from", _fail)

    with pytest.raises(RuntimeError, match="simulated network error"):
        spiller_connect.save_checkpoint_pl(df_new, name="ckpt_c4", storage="volume")

    loaded = spiller_connect.load_checkpoint_pl("ckpt_c4", eager=True, storage="volume")
    assert loaded.sort("id").equals(df_old.sort("id"))


def test_save_checkpoint_pl_local_unchanged_under_connect(spiller_connect):
    df = pl.DataFrame({"id": [1]})
    spiller_connect.save_checkpoint_pl(df, name="gold_local", storage="local")
    expected = spiller_connect.local_base_dir / "gold_local" / "data.parquet"
    assert expected.exists()


def test_load_checkpoint_pl_volume_under_connect_roundtrip(spiller_connect):
    df = pl.DataFrame({"id": [1, 2], "val": ["a", "b"]})
    spiller_connect.save_checkpoint_pl(df, name="rt_vol", storage="volume")
    loaded = spiller_connect.load_checkpoint_pl("rt_vol", eager=True, storage="volume")
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.sort("id").equals(df.sort("id"))


def test_load_checkpoint_pl_volume_raises_when_missing(spiller_connect):
    with pytest.raises(FileNotFoundError):
        spiller_connect.load_checkpoint_pl("does_not_exist", storage="volume")


def test_load_checkpoint_pl_volume_lazy_tracks_staging(spiller_connect):
    df = pl.DataFrame({"id": [1]})
    spiller_connect.save_checkpoint_pl(df, name="lazy_vol", storage="volume")
    lf = spiller_connect.load_checkpoint_pl("lazy_vol", eager=False, storage="volume")
    assert isinstance(lf, pl.LazyFrame)
    assert len(spiller_connect._active_local_dirs) == 1
    assert lf.collect().shape == (1, 1)


def test_list_checkpoints_volume_under_connect(spiller_connect, monkeypatch):
    df = pl.DataFrame({"id": [1]})
    spiller_connect.save_checkpoint_pl(df, name="ckpt_a", storage="volume")
    spiller_connect.save_checkpoint_pl(df, name="ckpt_b", storage="volume")

    real_listdir = os.listdir

    def guarded_listdir(path):
        if str(path).startswith(str(spiller_connect.volume_root)):
            raise AssertionError(f"os.listdir called on volume path: {path}")
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", guarded_listdir)

    names = spiller_connect.list_checkpoints(storage="volume")
    assert sorted(names) == ["ckpt_a", "ckpt_b"]


def test_list_checkpoints_local_under_connect(spiller_connect):
    df = pl.DataFrame({"id": [1]})
    spiller_connect.save_checkpoint_pl(df, name="local_ckpt", storage="local")
    names = spiller_connect.list_checkpoints(storage="local")
    assert names == ["local_ckpt"]


@pytest.mark.requires_pyspark
def test_load_checkpoint_spark_under_connect_roundtrip(spiller_connect, spark):
    spark_df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "txt"])
    spiller_connect.save_checkpoint_spark(spark_df, name="sp_ckpt")

    loaded = spiller_connect.load_checkpoint_spark("sp_ckpt")
    rows = loaded.sort("id").collect()
    assert [r["id"] for r in rows] == [1, 2]


def test_load_checkpoint_spark_raises_when_missing_under_connect(spiller_connect):
    with pytest.raises(FileNotFoundError):
        spiller_connect.load_checkpoint_spark("absent_ckpt")


@pytest.mark.requires_pyspark
def test_load_checkpoint_spark_under_connect_does_not_call_os_path_exists_on_volume(
    spiller_connect, spark, monkeypatch
):
    spark_df = spark.createDataFrame([(1,)], ["id"])
    spiller_connect.save_checkpoint_spark(spark_df, name="sp_ckpt2")

    real_exists = os.path.exists

    def guarded_exists(path):
        if str(path).startswith(str(spiller_connect.volume_root)):
            raise AssertionError(f"os.path.exists called on volume path: {path}")
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", guarded_exists)
    loaded = spiller_connect.load_checkpoint_spark("sp_ckpt2")
    assert loaded.count() == 1


# ──────────────────────────────────────────────────────────────────────────────
# I1: parallel uploads/downloads with exponential-backoff retry
# ──────────────────────────────────────────────────────────────────────────────


def test_retry_op_succeeds_after_transient_failures(monkeypatch):
    """_retry_op retries on transient SDK errors and returns the successful result."""
    from databricks.sdk.errors import ResourceExhausted

    monkeypatch.setattr("databricks_scaffold.core.time.sleep", lambda _: None)

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ResourceExhausted("rate limited")
        return "ok"

    result = _retry_op(flaky)
    assert result == "ok"
    assert calls["n"] == 3


def test_retry_op_raises_after_max_retries_exhausted(monkeypatch):
    """_retry_op raises the last exception when max retries are exhausted."""
    from databricks.sdk.errors import ResourceExhausted

    monkeypatch.setattr("databricks_scaffold.core.time.sleep", lambda _: None)

    def always_fails():
        raise ResourceExhausted("always rate limited")

    with pytest.raises(ResourceExhausted):
        _retry_op(always_fails, max_retries=3)


def test_retry_op_propagates_non_retryable_errors_immediately(monkeypatch):
    """_retry_op does not retry errors outside _RETRYABLE_SDK_ERRORS."""
    from databricks.sdk.errors import PermissionDenied

    monkeypatch.setattr("databricks_scaffold.core.time.sleep", lambda _: None)

    calls = {"n": 0}

    def one_shot():
        calls["n"] += 1
        raise PermissionDenied("forbidden")

    with pytest.raises(PermissionDenied):
        _retry_op(one_shot, max_retries=5)

    assert calls["n"] == 1


def test_upload_dir_retries_on_transient_error(spiller_connect, tmp_path, monkeypatch):
    """Single-file upload retries a 429 and ultimately succeeds."""
    from databricks.sdk.errors import ResourceExhausted

    monkeypatch.setattr("databricks_scaffold.core.time.sleep", lambda _: None)

    local_src = tmp_path / "src"
    local_src.mkdir()
    (local_src / "part-0.parquet").write_bytes(b"data")

    attempt_counts = {"n": 0}
    real_upload = spiller_connect._workspace_client.files.upload_from

    def flaky_upload(*args, **kwargs):
        attempt_counts["n"] += 1
        if attempt_counts["n"] < 3:
            raise ResourceExhausted("rate limited")
        return real_upload(*args, **kwargs)

    monkeypatch.setattr(spiller_connect._workspace_client.files, "upload_from", flaky_upload)

    remote = f"{spiller_connect.volume_root}/retry_upload"
    spiller_connect._upload_dir_to_volume(str(local_src), remote)

    assert attempt_counts["n"] == 3
    assert os.path.exists(f"{remote}/part-0.parquet")


def test_upload_dir_raises_after_exhausted_retries(spiller_connect, tmp_path, monkeypatch):
    """Upload raises after all retry attempts are exhausted."""
    from databricks.sdk.errors import ResourceExhausted

    monkeypatch.setattr("databricks_scaffold.core.time.sleep", lambda _: None)

    local_src = tmp_path / "src"
    local_src.mkdir()
    (local_src / "part-0.parquet").write_bytes(b"data")

    def always_fails(*args, **kwargs):
        raise ResourceExhausted("always fails")

    monkeypatch.setattr(spiller_connect._workspace_client.files, "upload_from", always_fails)

    remote = f"{spiller_connect.volume_root}/retry_exhausted"
    with pytest.raises(ResourceExhausted):
        spiller_connect._upload_dir_to_volume(str(local_src), remote)


def test_upload_dir_many_files_all_land(spiller_connect, tmp_path):
    """Parallelised upload correctly transfers all 10 parquet files without loss."""
    local_src = tmp_path / "src"
    local_src.mkdir()
    n = 10
    for i in range(n):
        (local_src / f"part-{i:04d}.parquet").write_bytes(f"data{i}".encode())

    remote = f"{spiller_connect.volume_root}/parallel_upload"
    spiller_connect._upload_dir_to_volume(str(local_src), remote)

    uploaded = sorted(f for f in os.listdir(remote) if f.endswith(".parquet"))
    assert len(uploaded) == n


def test_download_dir_many_files_all_land(spiller_connect, tmp_path):
    """Parallelised download correctly retrieves all 10 parquet files without loss."""
    remote = f"{spiller_connect.volume_root}/parallel_download"
    os.makedirs(remote)
    n = 10
    for i in range(n):
        with open(f"{remote}/part-{i:04d}.parquet", "wb") as f:
            f.write(f"data{i}".encode())

    local_dst = tmp_path / "dst"
    local_dst.mkdir()
    spiller_connect._download_volume_dir(remote, str(local_dst))

    downloaded = sorted(f for f in os.listdir(str(local_dst)) if f.endswith(".parquet"))
    assert len(downloaded) == n


# ──────────────────────────────────────────────────────────────────────────────
# I6: _DF_TYPES tuple covers ConnectDataFrame, not just SparkDataFrame
# ──────────────────────────────────────────────────────────────────────────────


def test_df_types_includes_connect_dataframe():
    """_DF_TYPES must contain ConnectDataFrame when pyspark.sql.connect is importable.
    If someone removes it from the tuple, this test fails immediately."""
    ConnectDataFrame = pytest.importorskip(
        "pyspark.sql.connect.dataframe",
        reason="pyspark.sql.connect not available",
    ).DataFrame
    assert ConnectDataFrame in _core._DF_TYPES


def test_save_checkpoint_spark_accepts_connect_dataframe(spiller_connect, monkeypatch):
    """A ConnectDataFrame instance must pass the isinstance guard.
    Existing tests use SparkDataFrame, which matches the first element of _DF_TYPES —
    this test exercises the ConnectDataFrame element specifically."""
    ConnectDataFrame = pytest.importorskip(
        "pyspark.sql.connect.dataframe",
        reason="pyspark.sql.connect not available",
    ).DataFrame

    # object.__new__ bypasses ConnectDataFrame.__init__ (which needs a gRPC plan/session)
    # giving us a real ConnectDataFrame instance that passes isinstance.
    connect_df = object.__new__(ConnectDataFrame)
    connect_df.coalesce = MagicMock(return_value=connect_df)
    connect_df.write = MagicMock()
    connect_df.write.parquet = MagicMock(side_effect=RuntimeError("write stubbed"))

    monkeypatch.setattr(spiller_connect, "_volume_mkdirs", lambda *a, **kw: None)
    monkeypatch.setattr(spiller_connect, "_volume_rmtree", lambda *a, **kw: None)

    with pytest.raises(RuntimeError, match="write stubbed"):
        spiller_connect.save_checkpoint_spark(connect_df, name="connect_ckpt")


def test_save_checkpoint_spark_rejects_non_dataframe(spiller_connect):
    """Non-DataFrame types must raise TypeError, not AttributeError or worse."""
    with pytest.raises(TypeError):
        spiller_connect.save_checkpoint_spark("not-a-df", name="bad_ckpt")


# ──────────────────────────────────────────────────────────────────────────────
# I8: helpful error message when Connect cluster is stopped at __init__
# ──────────────────────────────────────────────────────────────────────────────


def test_init_raises_helpful_error_when_connect_cluster_down(monkeypatch):
    """When spark.sql raises under Connect, __init__ re-raises with a clear hint."""
    from databricks_scaffold.core import VolumeSpiller

    mock_spark = MagicMock()
    mock_spark.sql.side_effect = Exception("UNAVAILABLE: cluster not running")
    monkeypatch.setattr(_core, "_is_databricks_connect", lambda _: True)

    with pytest.raises(RuntimeError, match="Is your Connect cluster started"):
        VolumeSpiller(mock_spark, "cat", "sch", "vol", is_dev=True)


def test_init_non_connect_error_propagates_unchanged(monkeypatch):
    """For non-Connect sessions, spark.sql errors propagate as-is — no wrapping."""
    from databricks_scaffold.core import VolumeSpiller

    original_error = ValueError("unexpected SQL error")
    mock_spark = MagicMock()
    mock_spark.sql.side_effect = original_error
    monkeypatch.setattr(_core, "_is_databricks_connect", lambda _: False)

    with pytest.raises(ValueError, match="unexpected SQL error"):
        VolumeSpiller(mock_spark, "cat", "sch", "vol", is_dev=True)
