from typing import Any


def _get_notebook_var(var_name: str) -> Any:
    """
    Fetches a variable directly from the Databricks notebook's interactive namespace.
    """
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        if ipy is not None and var_name in ipy.user_ns:
            return ipy.user_ns[var_name]
    except ImportError:
        pass
    return None


def _resolve_is_dev(explicit: bool | None) -> bool:
    """
    Resolves the is_dev flag using a three-tier priority:
    1. Explicit argument (if not None)
    2. IS_DEV variable from the notebook namespace (handles widget strings)
    3. Default: True (dev mode)
    """
    if explicit is not None:
        return explicit
    raw = _get_notebook_var("IS_DEV")
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() not in ("false", "f", "0", "no", "")
    return bool(raw)
