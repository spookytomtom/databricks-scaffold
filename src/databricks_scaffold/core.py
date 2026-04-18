import uuid
import shutil
import tempfile
import os
import glob
import functools
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
import getpass
from pathlib import Path
import re
from databricks_scaffold._internal import _resolve_is_dev

def _is_databricks_connect(spark) -> bool:
    """
    Detects whether the given SparkSession is a Databricks Connect session.

    Returns True only when databricks.connect is installed AND the session is an
    instance of DatabricksSession. Returns False for plain pyspark.sql.SparkSession
    (on-cluster or local) and when the databricks.connect package is unavailable.
    """
    try:
        from databricks.connect.session import DatabricksSession
    except ImportError:
        return False
    return isinstance(spark, DatabricksSession)

class VolumeSpiller:
    """
    A utility class to manage data spilling and checkpointing between PySpark and Polars
    using Databricks Unity Catalog Volumes and local driver storage.
    """
    def __init__(self, spark: SparkSession, catalog: str, schema: str, volume_name: str, is_dev: bool = None):
        """
        Initializes the VolumeSpiller, setting up paths and managing the underlying UC Volume.

        Args:
            spark (SparkSession): The active SparkSession. If None, gets or creates one.
            catalog (str): The Unity Catalog name.
            schema (str): The schema (database) name within the catalog.
            volume_name (str): The name of the volume to use for spilling.
            is_dev (bool, optional): If True, preserves data and uses CREATE IF NOT EXISTS.
                                     If False (Prod), drops and recreates the volume for a clean slate.
                                     If not provided, reads IS_DEV from the notebook namespace.
                                     Defaults to True if IS_DEV is not set anywhere.
        """
        self.spark = spark if spark else SparkSession.builder.getOrCreate()
        self.is_dev = _resolve_is_dev(is_dev)
        self.full_name = f"{catalog}.{schema}.{volume_name}"
        self.volume_root = f"/Volumes/{catalog}/{schema}/{volume_name}"
        user = getpass.getuser()
        self.local_base_dir = Path(f"/tmp/{user}/spill")
        self.local_base_dir.mkdir(parents=True, exist_ok=True)
        
        if is_dev:
            # In Dev, we just want to make sure it exists so we don't lose data
            self.spark.sql(f"CREATE VOLUME IF NOT EXISTS {self.full_name}")
        else:
            # In Prod, we want a clean slate, so we drop then create
            self.spark.sql(f"DROP VOLUME IF EXISTS {self.full_name}")
            self.spark.sql(f"CREATE VOLUME {self.full_name}")
            
        self._active_local_dirs: list[str] = []
        self._active_volume_dirs: list[str] = []
        self._is_connect = _is_databricks_connect(self.spark)
        self._w = None

    @property
    def _workspace(self):
        """Lazy WorkspaceClient accessor. Only built when first volume I/O happens under Connect."""
        if self._w is None:
            from databricks.sdk import WorkspaceClient
            self._w = WorkspaceClient()
        return self._w

    def _volume_mkdirs(self, volume_path: str) -> None:
        """Create a directory on the volume (idempotent). Routes via Files API under Connect."""
        if self._is_connect:
            self._workspace.files.create_directory(volume_path)
        else:
            os.makedirs(volume_path, exist_ok=True)

    def _volume_exists(self, volume_path: str) -> bool:
        """Check whether a directory exists on the volume."""
        if self._is_connect:
            try:
                self._workspace.files.get_directory_metadata(volume_path)
                return True
            except Exception:
                return False
        return os.path.exists(volume_path)

    def _volume_listdir(self, volume_path: str) -> list[str]:
        """List entry names in a volume directory. Returns [] for missing directory."""
        if self._is_connect:
            try:
                return [e.name for e in self._workspace.files.list_directory_contents(volume_path)]
            except Exception:
                return []
        if not os.path.exists(volume_path):
            return []
        return os.listdir(volume_path)

    def _volume_rmtree(self, volume_path: str) -> None:
        """
        Remove a volume directory and all its contents. Silent when missing.
        Under Connect, the Files API has no recursive delete — we walk contents,
        delete each file, then remove the now-empty directory. Nested subdirs are
        recursed into.
        """
        if not self._is_connect:
            shutil.rmtree(volume_path, ignore_errors=True)
            return

        try:
            entries = list(self._workspace.files.list_directory_contents(volume_path))
        except Exception:
            return  # directory doesn't exist — nothing to do

        for entry in entries:
            if entry.is_directory:
                self._volume_rmtree(entry.path)
            else:
                try:
                    self._workspace.files.delete(entry.path)
                except Exception:
                    pass
        try:
            self._workspace.files.delete_directory(volume_path)
        except Exception:
            pass

    def _download_volume_dir(self, volume_dir: str, local_dir: str) -> None:
        """
        Download every *.parquet file in volume_dir to local_dir. Non-parquet files
        are skipped for the same reason as _upload_dir_to_volume.
        """
        os.makedirs(local_dir, exist_ok=True)
        for entry in self._workspace.files.list_directory_contents(volume_dir):
            if entry.is_directory or not entry.name.endswith(".parquet"):
                continue
            dst = os.path.join(local_dir, entry.name)
            self._workspace.files.download_to(
                file_path=entry.path,
                local_path=dst,
                use_parallel=True,
            )

    def _upload_dir_to_volume(self, local_dir: str, volume_dir: str) -> None:
        """
        Upload every *.parquet file in local_dir to volume_dir. Non-parquet files
        (_SUCCESS, .crc) are skipped — they are Spark-side artifacts that Polars
        doesn't need and that the round-trip shouldn't propagate.
        """
        self._volume_mkdirs(volume_dir)
        for name in os.listdir(local_dir):
            if not name.endswith(".parquet"):
                continue
            src = os.path.join(local_dir, name)
            dst = f"{volume_dir}/{name}"
            self._workspace.files.upload_from(
                file_path=dst,
                source_path=src,
                overwrite=True,
                use_parallel=True,
            )

    def get_path(self, name: str) -> str:
        """
        Returns the absolute path for a named folder within the UC volume.

        Args:
            name (str): The name of the folder or checkpoint.

        Returns:
            str: The absolute path string.
        """
        return f"{self.volume_root}/{name.lstrip('/')}"
    
    def _resolve_path(self, name: str, storage: str):
        """
        Internal helper to resolve the base path based on the target storage tier.

        For 'volume' storage, directory creation is routed through _volume_mkdirs so
        it works under Databricks Connect (Files API) as well as on-cluster (os.makedirs).

        Args:
            name (str): The name of the checkpoint or folder.
            storage (str): The storage tier, either 'volume' or 'local'.

        Returns:
            tuple[str, str]: (resolved absolute path, storage type)

        Raises:
            ValueError: If storage is not 'volume' or 'local'.
        """
        if storage not in ("volume", "local"):
            raise ValueError("storage must be 'volume' or 'local'")

        if storage == "volume":
            path = self.get_path(name)
            self._volume_mkdirs(path)
        else:
            path = str(self.local_base_dir / name)
            os.makedirs(path, exist_ok=True)

        return path, storage
    
    def _prepare_polars_timestamps(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """
        Internal helper to fix timestamp precision and attach UTC timezones for Spark compatibility.

        Args:
            df (pl.DataFrame | pl.LazyFrame): The Polars DataFrame to process.

        Returns:
            pl.DataFrame | pl.LazyFrame: The processed Polars DataFrame with corrected timestamps.
        """
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        
        exprs = []
        for col_name, dtype in schema.items():
            if isinstance(dtype, pl.Datetime):
                expr = pl.col(col_name)
                modified = False
                
                # 1. Fix precision: Convert both nanoseconds and microseconds to milliseconds
                if dtype.time_unit in ("ns", "us"):
                    expr = expr.dt.cast_time_unit("ms")
                    modified = True
                    print(f"⏰ Auto-fix: Casting '{col_name}' from {dtype.time_unit} to ms.")
                
                # 2. Fix timezone: Add UTC to prevent timestamp_ntz in Spark
                if dtype.time_zone is None:
                    expr = expr.dt.replace_time_zone("UTC")
                    modified = True
                    print(f"🌍 Auto-fix: Adding UTC timezone to '{col_name}' to avoid timestamp_ntz.")
                    
                if modified:
                    exprs.append(expr)
                    
        if exprs:
            return df.with_columns(exprs)
        return df

    def list_checkpoints(self, storage: str = "volume") -> list[str]:
        """
        Lists all available checkpoints in the specified storage tier.

        Args:
            storage (str, optional): The storage tier, either 'volume' or 'local'. Defaults to 'volume'.

        Returns:
            list[str]: A sorted list of checkpoint directory names.

        Raises:
            ValueError: If the storage argument is not 'volume' or 'local'.
        """
        if storage == "volume":
            root = self.volume_root
            if not self._volume_exists(root):
                return []
            if self._is_connect:
                return sorted(
                    e.name
                    for e in self._workspace.files.list_directory_contents(root)
                    if e.is_directory
                )
            return sorted(
                name for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            )

        elif storage == "local":
            if not self.local_base_dir.exists():
                return []
            return sorted(
                p.name for p in self.local_base_dir.iterdir()
                if p.is_dir()
            )

        else:
            raise ValueError("storage must be 'volume' or 'local'")

    def save_checkpoint_pl(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        name: str,
        storage: str = "volume",
        compression: str = "auto"
    ) -> None:
        """
        Saves a Polars DataFrame checkpoint to either the UC Volume or driver-local /tmp.

        Under Databricks Connect with storage='volume', the parquet is written to a
        local /tmp staging file and then uploaded to the volume via the Files API.

        Args:
            df (pl.DataFrame | pl.LazyFrame): The Polars DataFrame to save.
            name (str): The name of the checkpoint.
            storage (str, optional): Target storage, 'volume' or 'local'. Defaults to 'volume'.
            compression (str, optional): 'auto', 'zstd', 'snappy', or 'uncompressed'.
                'auto' routes to 'zstd' for volume and 'snappy' for local. Defaults to 'auto'.

        Raises:
            TypeError: If df is not a Polars DataFrame/LazyFrame.
            ValueError: If name is empty or invalid.
        """
        df = self._prepare_polars_timestamps(df)

        if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(f"df must be pl.DataFrame or pl.LazyFrame, got {type(df).__name__}")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")

        if not re.match(r"^[\w\-]+$", name):
            raise ValueError(
                f"Invalid checkpoint name '{name}'. To prevent accidental deletions, "
                "names must only contain alphanumeric characters, underscores, or hyphens."
            )

        base_path, resolved_storage = self._resolve_path(name, storage)

        if compression == "auto":
            compression = "zstd" if resolved_storage == "volume" else "snappy"

        if resolved_storage == "volume":
            self._volume_rmtree(base_path)
            self._volume_mkdirs(base_path)
        else:
            shutil.rmtree(base_path, ignore_errors=True)
            os.makedirs(base_path, exist_ok=True)

        if resolved_storage == "volume" and self._is_connect:
            staging_dir = tempfile.mkdtemp(prefix="ckpt_pl_")
            try:
                local_file = os.path.join(staging_dir, "data.parquet")
                if isinstance(df, pl.LazyFrame):
                    df.sink_parquet(local_file, compression=compression)
                else:
                    df.write_parquet(local_file, compression=compression)
                self._upload_dir_to_volume(staging_dir, base_path)
            finally:
                shutil.rmtree(staging_dir, ignore_errors=True)
        else:
            target_path = f"{base_path}/data.parquet"
            if isinstance(df, pl.LazyFrame):
                df.sink_parquet(target_path, compression=compression)
            else:
                df.write_parquet(target_path, compression=compression)

        prefix = "⚡ Local" if resolved_storage == "local" else "✅ Volume"
        print(f"{prefix} checkpoint '{name}' written using {compression} compression.")

    def load_checkpoint_pl(
        self,
        name: str,
        eager: bool = True,
        storage: str = "volume"
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Loads a named Parquet checkpoint into a Polars DataFrame.

        Under Databricks Connect with storage='volume', the parquet files are
        downloaded to a local /tmp staging directory before Polars reads them.
        In eager mode the staging dir is cleaned immediately; in lazy mode it is
        added to _active_local_dirs so teardown() reclaims it.

        Args:
            name (str): The name of the checkpoint to load.
            eager (bool, optional): DataFrame if True, LazyFrame if False. Defaults to True.
            storage (str, optional): 'volume' or 'local'. Defaults to 'volume'.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
            ValueError: If name is invalid or storage is unsupported.
        """
        if not isinstance(name, str) or not re.match(r"^[\w\-]+$", name):
            raise ValueError(
                f"Invalid checkpoint name '{name}'. Names must only contain "
                "alphanumeric characters, underscores, or hyphens."
            )
        if storage not in ("volume", "local"):
            raise ValueError("storage must be 'volume' or 'local'")

        if storage == "volume":
            base_path = self.get_path(name)
            if not self._volume_exists(base_path):
                raise FileNotFoundError(f"Checkpoint '{name}' not found at {base_path}")

            if self._is_connect:
                staging_dir = tempfile.mkdtemp(prefix="ckpt_pl_load_")
                self._download_volume_dir(base_path, staging_dir)
                read_path = f"{staging_dir}/*.parquet"
                if eager:
                    try:
                        return pl.read_parquet(read_path)
                    finally:
                        shutil.rmtree(staging_dir, ignore_errors=True)
                else:
                    self._active_local_dirs.append(staging_dir)
                    return pl.scan_parquet(read_path)
            else:
                read_path = f"{base_path}/*.parquet"
                return pl.read_parquet(read_path) if eager else pl.scan_parquet(read_path)

        base_path = str(self.local_base_dir / name)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {base_path}")
        read_path = f"{base_path}/*.parquet"
        return pl.read_parquet(read_path) if eager else pl.scan_parquet(read_path)

    def save_checkpoint_spark(self, df: SparkDataFrame, name: str, optimize_files: bool = False) -> None:
        """
        Saves a PySpark DataFrame as a named Parquet checkpoint directory in the UC volume.

        Args:
            df (SparkDataFrame): The PySpark DataFrame to save.
            name (str): The name of the checkpoint directory.
            optimize_files (bool, optional): If True, coalesces the DataFrame to 2 partitions before writing. Defaults to False.

        Raises:
            TypeError: If arguments are of incorrect types.
        """
        if not isinstance(df, SparkDataFrame):
            raise TypeError(f"df must be a pyspark.sql.DataFrame, got {type(df).__name__}")

        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")

        if not isinstance(optimize_files, bool):
            raise TypeError(f"optimize_files must be bool, got {type(optimize_files).__name__}")

        # Strictly enforce directory name safety to prevent path traversal (e.g., "..", "/")
        if not re.match(r"^[\w\-]+$", name):
            raise ValueError(
                f"Invalid checkpoint name '{name}'. To prevent accidental deletions, "
                "names must only contain alphanumeric characters, underscores, or hyphens."
            )

        checkpoint_dir = self.get_path(name)

        if optimize_files:
            df = df.coalesce(2)
            
        # Hardcoding zstd since this explicitly saves to the Volume
        df.write.mode("overwrite").option("compression", "zstd").parquet(checkpoint_dir)
        print(f"✅ Spark checkpoint '{name}' written to UC Volume using zstd compression.")

    def load_checkpoint_spark(self, name: str) -> SparkDataFrame:
        """
        Loads a named Parquet checkpoint from the UC volume into a PySpark DataFrame.

        Args:
            name (str): The name of the checkpoint directory.

        Returns:
            SparkDataFrame: The loaded PySpark DataFrame.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
            ValueError: If name is invalid.
        """
        if not isinstance(name, str) or not re.match(r"^[\w\-]+$", name):
            raise ValueError(
                f"Invalid checkpoint name '{name}'. Names must only contain "
                "alphanumeric characters, underscores, or hyphens."
            )

        checkpoint_dir = self.get_path(name)
        if not self._volume_exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {checkpoint_dir}")
        return self.spark.read.parquet(checkpoint_dir)

    def teardown(self) -> None:
        """
        Cleans up local driver temp dirs and UC Volume state. In Dev mode, volume
        root is preserved; in Prod, it is dropped. Under Databricks Connect, volume
        cleanup goes through the Files API.
        """
        if self.local_base_dir.exists():
            shutil.rmtree(self.local_base_dir, ignore_errors=True)
            print(f"🧹 LOCAL CLEANUP: Cleared driver temp directory {self.local_base_dir}")

        for d in self._active_local_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._active_local_dirs.clear()

        for d in self._active_volume_dirs:
            self._volume_rmtree(d)
        self._active_volume_dirs.clear()

        if self.is_dev:
            print(f"🛠️ DEV MODE: Volume data preserved at {self.volume_root}")
        else:
            self.spark.sql(f"DROP VOLUME IF EXISTS {self.full_name}")
            print(f"🗑️ PROD MODE: Volume {self.full_name} dropped.")

    def spark_to_polars(self, df: SparkDataFrame, cleanup: bool = False, eager: bool = True, optimize_files: bool = False) -> pl.DataFrame | pl.LazyFrame:
        """
        Spills a PySpark DataFrame to the UC volume, then reads it back as Polars.

        Under Databricks Connect, the parquet files are additionally downloaded from
        the volume to a local /tmp directory before Polars reads them, because
        /Volumes/... is not mounted on the local driver.

        Args:
            df (SparkDataFrame): The input PySpark DataFrame.
            cleanup (bool, optional): If True and eager is True, deletes the temporary
                volume directory (and the local staging dir under Connect) after reading.
                Defaults to False.
            eager (bool, optional): If True, reads as a pl.DataFrame. If False, scans as a
                pl.LazyFrame. Defaults to True.
            optimize_files (bool, optional): If True, coalesces to 2 partitions before
                writing. Defaults to False.

        Returns:
            pl.DataFrame | pl.LazyFrame
        """
        run_id = uuid.uuid4().hex
        volume_temp_dir = self.get_path(f"spill_sp_pl_{run_id}")
        track_volume = not (cleanup and eager)
        if track_volume:
            self._active_volume_dirs.append(volume_temp_dir)

        local_staging_dir = None
        try:
            if optimize_files:
                df = df.coalesce(2)

            df.write.mode("overwrite").option("compression", "zstd").parquet(volume_temp_dir)

            if self._is_connect:
                local_staging_dir = tempfile.mkdtemp(prefix="spill_sp_pl_")
                self._download_volume_dir(volume_temp_dir, local_staging_dir)
                read_path = f"{local_staging_dir}/*.parquet"
            else:
                read_path = f"{volume_temp_dir}/*.parquet"

            if eager:
                result = pl.read_parquet(read_path)
                if self._is_connect and local_staging_dir is not None:
                    shutil.rmtree(local_staging_dir, ignore_errors=True)
                    local_staging_dir = None
                return result
            else:
                if self._is_connect and local_staging_dir is not None:
                    self._active_local_dirs.append(local_staging_dir)
                    local_staging_dir = None  # ownership transferred to teardown
                return pl.scan_parquet(read_path)

        finally:
            if cleanup and eager:
                self._volume_rmtree(volume_temp_dir)
            if local_staging_dir is not None:
                shutil.rmtree(local_staging_dir, ignore_errors=True)

    def polars_to_spark(self, df: pl.DataFrame | pl.LazyFrame) -> SparkDataFrame:
        """
        Spills a Polars DataFrame to the UC volume and reads it back as a PySpark DataFrame.

        Under Databricks Connect, Polars writes to a local /tmp staging file first; the
        file is then uploaded to the volume via the Files API, and Spark reads the volume
        directory. On-cluster, Polars writes directly to the FUSE-mounted volume path.

        Args:
            df (pl.DataFrame | pl.LazyFrame): The input Polars DataFrame or LazyFrame.

        Returns:
            SparkDataFrame
        """
        run_id = uuid.uuid4().hex
        volume_temp_dir = self.get_path(f"spill_pl_sp_{run_id}")
        self._active_volume_dirs.append(volume_temp_dir)

        df = self._prepare_polars_timestamps(df)

        if self._is_connect:
            local_staging_dir = tempfile.mkdtemp(prefix="spill_pl_sp_")
            try:
                local_file = os.path.join(local_staging_dir, "part-0.parquet")
                if isinstance(df, pl.LazyFrame):
                    df.sink_parquet(local_file, compression="zstd")
                else:
                    df.write_parquet(local_file, compression="zstd")
                self._upload_dir_to_volume(local_staging_dir, volume_temp_dir)
            finally:
                shutil.rmtree(local_staging_dir, ignore_errors=True)
        else:
            os.makedirs(volume_temp_dir, exist_ok=True)
            volume_file = f"{volume_temp_dir}/part-0.parquet"
            if isinstance(df, pl.LazyFrame):
                df.sink_parquet(volume_file, compression="zstd")
            else:
                df.write_parquet(volume_file, compression="zstd")

        return self.spark.read.parquet(volume_temp_dir)