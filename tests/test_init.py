import subprocess
import sys


def test_importing_package_does_not_import_pyspark():
    """
    Regression: databricks_scaffold must be importable without pulling pyspark
    into sys.modules. This keeps the package usable in pure-Polars environments.
    """
    code = (
        "import sys; "
        "import databricks_scaffold; "
        "assert 'pyspark' not in sys.modules, "
        "'pyspark was imported at package import time'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
