import uuid
import shutil
import os
import glob
import functools
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
import getpass
from pathlib import Path

class VolumeSpiller:
    """
    A utility class to manage data spilling and checkpointing between PySpark and Polars
    using Databricks Unity Catalog Volumes and local driver storage.
    """
    def __init__(self, spark: SparkSession, catalog: str, schema: str, volume_name: str, is_dev: bool = False):
        """
        Initializes the VolumeSpiller, setting up paths and managing the underlying UC Volume.

        Args:
            spark (SparkSession): The active SparkSession. If None, gets or creates one.
            catalog (str): The Unity Catalog name.
            schema (str): The schema (database) name within the catalog.
            volume_name (str): The name of the volume to use for spilling.
            is_dev (bool, optional): If True, preserves data and uses CREATE IF NOT EXISTS. 
                                     If False (Prod), drops and recreates the volume for a clean slate. Defaults to False.
        """
        self.spark = spark if spark else SparkSession.builder.getOrCreate()
        self.is_dev = is_dev
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
            
        self._active_temp_dirs = []

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

        Args:
            name (str): The name of the checkpoint or folder.
            storage (str): The storage tier, either 'volume' or 'local'.

        Returns:
            tuple[str, str]: A tuple containing the resolved absolute path and the storage type.

        Raises:
            ValueError: If the storage argument is not 'volume' or 'local'.
        """
        if storage not in ("volume", "local"):
            raise ValueError("storage must be 'volume' or 'local'")

        if storage == "volume":
            path = self.get_path(name)
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
            if not os.path.exists(root):
                return []
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

        Args:
            df (pl.DataFrame | pl.LazyFrame): The Polars DataFrame to save.
            name (str): The name of the checkpoint.
            storage (str, optional): Target storage, either 'volume' or 'local'. Defaults to 'volume'.
            compression (str, optional): Compression algorithm ('auto', 'zstd', 'snappy', 'uncompressed').
                                         'auto' routes to 'zstd' for volume and 'snappy' for local. Defaults to 'auto'.

        Raises:
            TypeError: If df is not a Polars DataFrame/LazyFrame.
            ValueError: If name is empty or invalid.
        """
        # AUTO-FIX TIMESTAMPS
        df = self._prepare_polars_timestamps(df)

        if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(f"df must be pl.DataFrame or pl.LazyFrame, got {type(df).__name__}")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")

        base_path, resolved_storage = self._resolve_path(name, storage)
        target_path = f"{base_path}/data.parquet"

        # AUTO-ROUTE COMPRESSION
        if compression == "auto":
            compression = "zstd" if resolved_storage == "volume" else "snappy"

        if isinstance(df, pl.LazyFrame):
            df.sink_parquet(target_path, compression=compression, engine="streaming")
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

        Args:
            name (str): The name of the checkpoint to load.
            eager (bool, optional): If True, returns a DataFrame. If False, returns a LazyFrame. Defaults to True.
            storage (str, optional): Storage location, either 'volume' or 'local'. Defaults to 'volume'.

        Returns:
            pl.DataFrame | pl.LazyFrame: The loaded Polars DataFrame or LazyFrame.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
        """

        base_path, _ = self._resolve_path(name, storage)

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {base_path}")

        glob_path = f"{base_path}/*.parquet"

        return pl.read_parquet(glob_path) if eager else pl.scan_parquet(glob_path)

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
        """
        checkpoint_dir = self.get_path(name)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {checkpoint_dir}")
        return self.spark.read.parquet(checkpoint_dir)

    def teardown(self) -> None:
        """
        Cleans up the initialized volume. In Dev mode, data is preserved for inspection.
        In Prod mode, the volume and its contents are dropped.
        """
        if self.is_dev:
            print(f"🛠️ DEV MODE: Data preserved at {self.volume_root}")
        else:
            self.spark.sql(f"DROP VOLUME IF EXISTS {self.full_name}")
            print(f"🗑️ PROD MODE: Volume {self.full_name} dropped.")

    def spark_to_polars(self, df: SparkDataFrame, cleanup: bool = False, eager: bool = True, optimize_files: bool = False) -> pl.DataFrame | pl.LazyFrame:
        """
        Spills a PySpark DataFrame to the UC volume and reads it back as a Polars DataFrame.

        Args:
            df (SparkDataFrame): The input PySpark DataFrame.
            cleanup (bool, optional): If True and eager is True, deletes the temporary volume directory after reading. Defaults to False.
            eager (bool, optional): If True, reads as a pl.DataFrame. If False, scans as a pl.LazyFrame. Defaults to True.
            optimize_files (bool, optional): If True, coalesces to 2 partitions before writing. Defaults to False.

        Returns:
            pl.DataFrame | pl.LazyFrame: The resulting Polars DataFrame or LazyFrame.
        """
        run_id = uuid.uuid4().hex
        temp_dir = self.get_path(f"spill_sp_pl_{run_id}")
        
        if not (cleanup and eager):
            self._active_temp_dirs.append(temp_dir)

        try:
            if optimize_files:
                df = df.coalesce(2) 

            # Hardcoding zstd since temp_dir routes to the Volume
            df.write.mode("overwrite").option("compression", "zstd").parquet(temp_dir)
            glob_path = f"{temp_dir}/*.parquet"
            
            if eager:
                return pl.read_parquet(glob_path)
            else:
                return pl.scan_parquet(glob_path)

        finally:
            if cleanup and eager:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def polars_to_spark(self, df: pl.DataFrame | pl.LazyFrame, cleanup: bool = False) -> SparkDataFrame:
        """
        Spills a Polars DataFrame to the UC volume and reads it back as a PySpark DataFrame.

        Args:
            df (pl.DataFrame | pl.LazyFrame): The input Polars DataFrame or LazyFrame.
            cleanup (bool, optional): If True, triggers a collection and deletes the temporary volume directory. Defaults to False.

        Returns:
            SparkDataFrame: The resulting PySpark DataFrame.
        """
        run_id = uuid.uuid4().hex
        temp_dir = self.get_path(f"spill_pl_sp_{run_id}")
        
        if not cleanup: 
            self._active_temp_dirs.append(temp_dir)

        # Timestamp fix applies here
        df = self._prepare_polars_timestamps(df)

        try:
            os.makedirs(temp_dir, exist_ok=True)
            file_path = f"{temp_dir}/part-0.parquet"
            
            # Hardcoding zstd since temp_dir routes to the Volume
            if isinstance(df, pl.LazyFrame):
                df.sink_parquet(file_path, compression="zstd")
            else:
                df.write_parquet(file_path, compression="zstd")

            spark_df = self.spark.read.parquet(temp_dir)
            
            if cleanup:
                spark_df.collect()
                
            return spark_df

        finally:
            if cleanup:
                shutil.rmtree(temp_dir, ignore_errors=True)