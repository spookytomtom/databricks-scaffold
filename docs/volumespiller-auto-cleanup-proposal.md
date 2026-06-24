# Proposal: opt-in automatic volume cleanup on error (`drop_on_error`)

## Summary

Add an opt-in `drop_on_error` flag to `VolumeSpiller` so that in prod mode
(`is_dev=False`) the Unity Catalog Volume is dropped automatically if an
uncaught error occurs anywhere in a notebook — **without** requiring users to
wrap every cell in `try/finally`. The flag defaults to `False`, so existing
behavior is unchanged.

## Current behavior

- `VolumeSpiller.__init__` registers `atexit.register(self.teardown)` **only when
  `is_dev=True`**.
- `teardown()` drops the volume **only when `is_dev=False`** (in dev it preserves
  the volume).

These two never combine into automatic cleanup. The net effect is that **prod has
no automatic cleanup at all** — the documented production pattern is a manual
`try/finally` + `teardown()`.

| Mode | `atexit` registered? | `teardown()` drops volume? | Auto-cleanup on error? |
|------|----------------------|----------------------------|------------------------|
| dev (`is_dev=True`)   | Yes | No (preserves) | n/a — volume kept by design |
| prod (`is_dev=False`) | No  | Yes            | **No** — requires manual `try/finally` |

## Problem

Wrapping a whole notebook in `try/finally` is awkward in practice:

- A `with`/`try` block cannot span multiple notebook cells — it lives in one cell.
- Indenting all working code under `try:` causes the Databricks notebook linter to
  flag the cell with a persistent yellow underline that cannot be disabled.
- It is easy to forget, which leaves orphaned volumes that keep costing storage in
  prod.

## Goal

Provide a one-time, opt-in way to guarantee prod volume cleanup on **any** uncaught
error, in **both** interactive notebooks and scheduled Jobs, with zero per-cell
boilerplate.

## Proposed API

Add a constructor flag (default `False`, fully backwards compatible):

```python
VolumeSpiller(
    spark,
    catalog,
    schema,
    volume_name,
    is_dev=None,
    workspace_client=None,
    drop_on_error=False,   # new
)
```

When `drop_on_error=True`, the instance installs cleanup hooks at construction
(works in both dev and prod modes — see "Implementation deviation" below):

1. An IPython custom exception handler (when running under IPython) that calls
   `teardown()` on any uncaught cell error, then re-shows the traceback so the run
   still fails.
2. `atexit.register(self.teardown)` as a backstop for clean process exit (Jobs) and
   kernel shutdown.

In dev mode the flag is a no-op — the volume is preserved by design, and staging
dirs are left intact for debugging.

## Sketch implementation (`core.py`)

In `__init__`, after the existing setup (and after `self._torn_down = False`):

```python
self._drop_on_error = drop_on_error
if drop_on_error and not self.is_dev:
    self._install_error_hooks()
```

New helper method:

```python
def _install_error_hooks(self) -> None:
    """Prod-only: drop the volume on any uncaught notebook error or at process exit.

    Installs two complementary hooks:
      * an IPython custom-exception handler that fires on any uncaught cell error
        (works interactively and in Jobs, while the Spark session is still alive);
      * an atexit handler as a backstop for clean process / kernel shutdown.

    teardown() is idempotent (guarded by self._torn_down), so multiple trigger
    paths are safe.
    """
    # Backstop: clean process / kernel exit.
    atexit.register(self.teardown)

    # Immediate: any uncaught cell exception (interactive + jobs).
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        ip = None

    if ip is not None:
        spiller = self

        def _cleanup_on_error(shell, etype, evalue, tb, tb_offset=None):
            try:
                spiller.teardown()
            finally:
                # Re-show the traceback so the cell / job still reports failure.
                shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

        ip.set_custom_exc((Exception,), _cleanup_on_error)
```

No change to `teardown()` is required — it is already idempotent via the
`_torn_down` guard.

## Proposed behavior matrix (with `drop_on_error=True`)

| Mode | On uncaught cell error | On process / kernel exit | On success |
|------|------------------------|--------------------------|------------|
| dev  | no-op (volume preserved) | `atexit` preserves volume | volume preserved |
| prod | `teardown()` drops volume, traceback still shown | `atexit` drops volume (backstop) | drops via `atexit` / explicit `teardown()` |

## Risks and considerations

- **`set_custom_exc` availability.** Databricks Python notebooks run on an IPython
  kernel (DBR 11+), so `get_ipython()` is present. The guarded import plus the
  `atexit` backstop cover environments where it is not.
- **Spark session at `atexit`.** In a Job, the Spark session may already be shutting
  down when `atexit` handlers run, so `DROP VOLUME` there can fail. The IPython hook
  (which runs while the session is alive) is the primary path; `atexit` is
  best-effort.
- **Do not swallow errors.** The handler re-shows the traceback, so the cell / job
  status still reflects failure.
- **Single handler slot.** `set_custom_exc` is global to the shell; if multiple
  spillers register, the last one wins. If multi-instance support is desired, a
  module-level registry of live spillers could fan out cleanup to all of them.
- **Idempotency.** The existing `_torn_down` guard already prevents a double drop.

## Implementation deviation

The shipped implementation (commit `baa5308` and later) extends `drop_on_error`
to **both** dev and prod modes, not just prod. When `drop_on_error=True`:

- In **prod**: drops the volume on error/exit (as proposed above).
- In **dev**: also drops the volume on error/exit — overriding the default
  dev-preserve behavior. This ensures a clean slate after a failed dev run.

The `teardown()` method checks `self.is_dev and not self._drop_on_error` to
decide preservation; when `drop_on_error` is set, the volume is always dropped.

## Backwards compatibility

Default `drop_on_error=False` means no behavior change for existing users. The new
hooks are installed only when a user explicitly opts in **and** is in prod mode.

## Suggested tests (mirroring `tests/test_core_connect.py` style)

- `drop_on_error=True, is_dev=False`: simulate an exception through a mocked
  `get_ipython().set_custom_exc` handler and assert `teardown()` is called and the
  traceback is re-shown.
- `drop_on_error=True, is_dev=False`: assert `atexit.register` is called with
  `teardown`.
- `drop_on_error=True, is_dev=True`: assert **no** hooks are installed (no
  `atexit`, no `set_custom_exc`).
- `drop_on_error=False`: assert no hooks are installed.
- Idempotency: invoking the handler twice drops the volume only once
  (`_torn_down`).
- No IPython available: `_install_error_hooks` still registers `atexit` and does not
  raise.

## Docs

Add a short "Automatic prod cleanup" subsection to the README Production pattern:

```python
# Prod: volume is auto-dropped on any error or at exit — no try/finally needed
spill = VolumeSpiller(spark, "main", "default", "etl_spill", drop_on_error=True)
```

## Notebook workaround available today (no library change)

Until a flag exists, the same effect can be achieved per-notebook by installing the
hooks manually in the init cell:

```python
import atexit
from IPython import get_ipython
from databricks_scaffold import VolumeSpiller

spill = VolumeSpiller(spark, CATALOG, SCHEMA, VOLUME, is_dev=IS_DEV)

def _spill_cleanup_on_error(shell, etype, evalue, tb, tb_offset=None):
    try:
        if not spill.is_dev:
            spill.teardown()
    finally:
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

get_ipython().set_custom_exc((Exception,), _spill_cleanup_on_error)

if not spill.is_dev:
    atexit.register(spill.teardown)
```

Folding this into the constructor via `drop_on_error` removes the boilerplate and
the linter-underline problem for all users.