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
    def __init__(self, spark: SparkSession, catalog: str, schema: str, volume_name: str, is_dev: bool = False):
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
        """Returns the absolute path for a named folder in the volume."""
        return f"{self.volume_root}/{name.lstrip('/')}"
    
    def _resolve_path(self, name: str, storage: str):
        if storage not in ("volume", "local"):
            raise ValueError("storage must be 'volume' or 'local'")

        if storage == "volume":
            path = self.get_path(name)
        else:
            path = str(self.local_base_dir / name)

        os.makedirs(path, exist_ok=True)
        return path, storage
    
    def _fix_ns_precision(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Internal helper to automatically cast ns timestamps to ms."""
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        ns_cols = [col for col, dtype in schema.items() if dtype == pl.Datetime("ns")]
        
        if ns_cols:
            print(f"⏰ Auto-fix: Casting {ns_cols} from nanoseconds (ns) to milliseconds (ms) for Spark/Parquet compatibility.")
            return df.with_columns([pl.col(c).dt.cast_time_unit("ms") for c in ns_cols])
        return df

    def list_checkpoints(self, storage: str = "volume") -> list[str]:
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
        compression: str = "zstd"
    ):
        """
        Saves a Polars checkpoint either to UC Volume or driver-local /tmp.

        storage:
            "volume" (default)
            "local"
        """

        # AUTO-FIX APPLIED HERE
        df = self._fix_ns_precision(df)

        if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(f"df must be pl.DataFrame or pl.LazyFrame, got {type(df).__name__}")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")

        base_path, resolved_storage = self._resolve_path(name, storage)
        target_path = f"{base_path}/data.parquet"

        if isinstance(df, pl.LazyFrame):
            df.sink_parquet(target_path, compression=compression, engine="streaming")
        else:
            df.write_parquet(target_path, compression=compression)

        prefix = "⚡ Local" if resolved_storage == "local" else "✅ Volume"
        print(f"{prefix} checkpoint '{name}' written.")

    def load_checkpoint_pl(
        self,
        name: str,
        eager: bool = True,
        storage: str = "volume"
    ) -> pl.DataFrame | pl.LazyFrame:

        base_path, _ = self._resolve_path(name, storage)

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {base_path}")

        glob_path = f"{base_path}/*.parquet"

        return pl.read_parquet(glob_path) if eager else pl.scan_parquet(glob_path)

    def save_checkpoint_spark(self, df: SparkDataFrame, name: str, optimize_files: bool = False):
        """
        Saves a Spark DataFrame as a named checkpoint (directory).
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
        df.write.mode("overwrite").option("compression", "zstd").parquet(checkpoint_dir)
        print(f"✅ Spark checkpoint '{name}' written to UC Volume.")

    def load_checkpoint_spark(self, name: str) -> SparkDataFrame:
        """
        Loads a named checkpoint into Spark DataFrame.
        """
        checkpoint_dir = self.get_path(name)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {checkpoint_dir}")
        return self.spark.read.parquet(checkpoint_dir)

    def teardown(self):
        """Cleans up. In Dev, we keep the data for inspection."""
        if self.is_dev:
            print(f"🛠️ DEV MODE: Data preserved at {self.volume_root}")
        else:
            self.spark.sql(f"DROP VOLUME IF EXISTS {self.full_name}")
            print(f"🗑️ PROD MODE: Volume {self.full_name} dropped.")

    def spark_to_polars(self, df: SparkDataFrame, cleanup: bool = False, eager: bool = True, optimize_files: bool = False) -> pl.DataFrame | pl.LazyFrame:
        run_id = uuid.uuid4().hex
        temp_dir = self.get_path(f"spill_sp_pl_{run_id}")
        
        if not (cleanup and eager):
            self._active_temp_dirs.append(temp_dir)

        try:
            if optimize_files:
                df = df.coalesce(2) 

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
        run_id = uuid.uuid4().hex
        temp_dir = self.get_path(f"spill_pl_sp_{run_id}")
        
        if not cleanup: 
            self._active_temp_dirs.append(temp_dir)

        # Timestamp fix applies here
        df = self._fix_ns_precision(df)

        # Use collect_schema to get schema for both DataFrame and LazyFrame
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema

        try:
            # Ensure the directory exists
            os.makedirs(temp_dir, exist_ok=True)
            file_path = f"{temp_dir}/part-0.parquet"
            
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