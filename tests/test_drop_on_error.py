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


def test_teardown_drops_volume_in_dev_with_drop_on_error(monkeypatch):
    """is_dev=True + drop_on_error=True → teardown drops the volume."""
    monkeypatch.setattr(atexit, "register", lambda f: None)
    monkeypatch.setattr(_core, "get_ipython", None)

    mock_spark = MagicMock()
    spiller = VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=True, drop_on_error=True,
    )
    spiller.teardown()

    drop_calls = [
        c for c in mock_spark.sql.call_args_list
        if "DROP VOLUME" in str(c)
    ]
    assert len(drop_calls) == 1


def test_teardown_preserves_volume_in_dev_without_drop_on_error(monkeypatch):
    """is_dev=True + drop_on_error=False → teardown preserves the volume (regression)."""
    monkeypatch.setattr(atexit, "register", lambda f: None)
    monkeypatch.setattr(_core, "get_ipython", None)

    mock_spark = MagicMock()
    spiller = VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=True, drop_on_error=False,
    )
    spiller.teardown()

    drop_calls = [
        c for c in mock_spark.sql.call_args_list
        if "DROP VOLUME" in str(c)
    ]
    assert len(drop_calls) == 0


def test_atexit_registered_when_prod_and_drop_on_error(monkeypatch):
    """is_dev=False + drop_on_error=True → atexit.register called with teardown."""
    mock_register = MagicMock()
    monkeypatch.setattr(atexit, "register", mock_register)
    monkeypatch.setattr(_core, "get_ipython", None)

    mock_spark = MagicMock()
    spiller = VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=False, drop_on_error=True,
    )

    mock_register.assert_called_with(spiller.teardown)


def test_atexit_not_double_registered_when_dev_and_drop_on_error(monkeypatch):
    """is_dev=True + drop_on_error=True → atexit registered once (not twice).

    Existing __init__ line 'if self.is_dev: atexit.register(...)' already
    registers. _install_error_hooks must skip the atexit call in dev mode
    to avoid a redundant second registration.
    """
    mock_register = MagicMock()
    monkeypatch.setattr(atexit, "register", mock_register)
    monkeypatch.setattr(_core, "get_ipython", None)

    mock_spark = MagicMock()
    spiller = VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=True, drop_on_error=True,
    )

    assert mock_register.call_count == 1
    mock_register.assert_called_with(spiller.teardown)
