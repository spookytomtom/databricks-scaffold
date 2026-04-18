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
