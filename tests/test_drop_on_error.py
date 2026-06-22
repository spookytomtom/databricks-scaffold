import atexit
from unittest.mock import MagicMock

from databricks_scaffold import core as _core
from databricks_scaffold.core import VolumeSpiller


def test_drop_on_error_defaults_to_false():
    """The new parameter defaults to False (backwards compatible)."""
    mock_spark = MagicMock()
    spiller = VolumeSpiller(
        mock_spark, "main", "default", "test_vol", is_dev=True
    )
    assert spiller._drop_on_error is False


def test_drop_on_error_true_is_stored():
    """When drop_on_error=True is passed, the attribute is stored."""
    mock_spark = MagicMock()
    spiller = VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=True, drop_on_error=True,
    )
    assert spiller._drop_on_error is True
