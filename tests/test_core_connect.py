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
