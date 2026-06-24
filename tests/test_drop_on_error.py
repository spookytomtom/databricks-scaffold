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


def test_ipython_hook_installed_when_available(monkeypatch):
    """When get_ipython() returns a shell, set_custom_exc is called with (Exception,) and a callable handler."""
    mock_register = MagicMock()
    monkeypatch.setattr(atexit, "register", mock_register)

    mock_shell = MagicMock()
    monkeypatch.setattr(_core, "get_ipython", lambda: mock_shell)

    mock_spark = MagicMock()
    VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=False, drop_on_error=True,
    )

    mock_shell.set_custom_exc.assert_called_once()
    call_args = mock_shell.set_custom_exc.call_args
    assert call_args[0][0] == (Exception,)
    handler = call_args[0][1]
    assert callable(handler)


def test_no_ipython_available_no_raise(monkeypatch):
    """When get_ipython is None (IPython not installed), only atexit registers. No raise."""
    mock_register = MagicMock()
    monkeypatch.setattr(atexit, "register", mock_register)
    monkeypatch.setattr(_core, "get_ipython", None)

    mock_spark = MagicMock()
    spiller = VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=False, drop_on_error=True,
    )

    mock_register.assert_called_with(spiller.teardown)


def test_ipython_returns_none_no_set_custom_exc(monkeypatch):
    """When get_ipython() returns None (IPython installed, no kernel), no set_custom_exc call."""
    mock_register = MagicMock()
    monkeypatch.setattr(atexit, "register", mock_register)
    monkeypatch.setattr(_core, "get_ipython", lambda: None)

    # get_ipython returns None, so set_custom_exc should never be called
    # on any shell. We just verify no exception is raised.
    mock_spark = MagicMock()
    VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=False, drop_on_error=True,
    )


def test_handler_calls_teardown_and_reraises(monkeypatch):
    """The IPython handler calls teardown() then re-shows the traceback."""
    mock_register = MagicMock()
    monkeypatch.setattr(atexit, "register", mock_register)

    mock_shell = MagicMock()
    monkeypatch.setattr(_core, "get_ipython", lambda: mock_shell)

    mock_spark = MagicMock()
    spiller = VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=False, drop_on_error=True,
    )

    handler = mock_shell.set_custom_exc.call_args[0][1]

    etype, evalue, tb = ValueError, ValueError("test error"), None
    handler(mock_shell, etype, evalue, tb)

    assert spiller._torn_down is True
    mock_shell.showtraceback.assert_called_once_with(
        (etype, evalue, tb), tb_offset=None
    )


def test_no_hooks_when_drop_on_error_false(monkeypatch):
    """drop_on_error=False → no atexit registration (beyond existing is_dev logic), no set_custom_exc."""
    mock_register = MagicMock()
    monkeypatch.setattr(atexit, "register", mock_register)

    mock_shell = MagicMock()
    monkeypatch.setattr(_core, "get_ipython", lambda: mock_shell)

    mock_spark = MagicMock()
    VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=False, drop_on_error=False,
    )

    # is_dev=False → existing __init__ does NOT register atexit
    # drop_on_error=False → _install_error_hooks NOT called
    mock_register.assert_not_called()
    mock_shell.set_custom_exc.assert_not_called()


def test_handler_idempotent_fired_twice_drops_once(monkeypatch):
    """Handler invoked twice → teardown runs once (_torn_down guard), showtraceback both times."""
    mock_register = MagicMock()
    monkeypatch.setattr(atexit, "register", mock_register)

    mock_shell = MagicMock()
    monkeypatch.setattr(_core, "get_ipython", lambda: mock_shell)

    mock_spark = MagicMock()
    VolumeSpiller(
        mock_spark, "main", "default", "test_vol",
        is_dev=False, drop_on_error=True,
    )

    # Clear construction SQL calls so we can isolate teardown's DROP VOLUME
    mock_spark.sql.reset_mock()

    handler = mock_shell.set_custom_exc.call_args[0][1]

    handler(mock_shell, ValueError, ValueError("first"), None)
    handler(mock_shell, ValueError, ValueError("second"), None)

    drop_calls = [
        c for c in mock_spark.sql.call_args_list
        if "DROP VOLUME" in str(c)
    ]
    assert len(drop_calls) == 1
    assert mock_shell.showtraceback.call_count == 2
