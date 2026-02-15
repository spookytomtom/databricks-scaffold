# databricks-scaffold
Productivity booster functions for using polars on Databricks. Ease the pain of working with small data on Databricks. Single node rebellion 🔥

Helps the process of switching back and forth between PySpark Dataframe and polars Dataframe and _more_.

Features:
- VolumeSpiller: 
  - Utilize Unity Catalog Volume as a spillage bucket making the switch between pyspark and polars faster or just to save checkpoint parquet files.
  - Utilize the driver node file system as a temporary scan / sink storage for polars.
- adds utility functions missing from pyspark, such as 
  - glimpse() - soon, 
  - keep_duplicates(), 
  - frame_shape(), 
  - clean_column_names() 
  - etc.

# Something that works, but...
As of PySpark 4.0 the DataFrame object has a [toArrow](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toArrow.html#pyspark.sql.DataFrame.toArrow)() method which helps the conversion to polars. 
Polars can utilize the [from_arrow](https://docs.pola.rs/api/python/stable/reference/api/polars.from_arrow.html)() method enabling zero-copy exchange of DataFrame.

# ... but
Read this part from PySpark:
`This method should only be used if the resulting PyArrow pyarrow.Table is expected to be small, as all the data is loaded into the driver's memory.`\
The catch is that when using a simple Serverless the PySpark DataFrame is stored on different nodes across the cluster. This means spark needs to collect it into the driver memory which takes network time and can produce memory spikes crashing the job.

One workaround is to load the data in batches, which is even slower.

# Something else that works, but...
Polars [scan_delta](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_delta.html)() works, but you need credentials and you can't yet write so not 100% solved process.

# Usage examples
Install it:
~~~
%pip install git+https://github.com/spookytomtom/databricks-scaffold.git
~~~

## 1. 🥪The "Polars Sandwich" (Spark ➡ Polars ➡ Spark)
The most common use case: taking a distributed Spark DataFrame, bringing it to the driver as Polars for fast single-node processing, and sending it back.
~~~
from databricks_scaffold import VolumeSpiller

# Initialize (Pro-tip: Use is_dev=True to keep data for inspection after run)
spill = VolumeSpiller(catalog="main", schema="default", volume_name="spill_vol", is_dev=True)

# 1. Spill Spark DataFrame to Volume and load as Polars
# This handles the I/O automatically
pl_df = spill.spark_to_polars(spark_df, optimize_files=True)

# 2. Do your heavy lifting in Polars (Standard API)
pl_result = (
    pl_df.filter(pl.col("status") == "active")
    .group_by("region")
    .agg(pl.sum("sales"))
)

# 3. Push back to Spark
# Note: This automatically fixes nanosecond timestamp issues compatible with Spark!
final_spark_df = spill.polars_to_spark(pl_result, cleanup=True)

final_spark_df.display()

# Clean up the volume when done
spill.teardown()
~~~
## 2. Checkpointing (Save & Load)
Stop recalculating the same data. Save intermediate states to the Unity Catalog Volume or the local driver disk.
### 💾 Saving to Unity Catalog (Persistent if configured)
Data persists across cluster restarts.
~~~
# Save a Polars DataFrame
spill.save_checkpoint_pl(pl_df, name="step_1_cleaned", storage="volume")

# Save a Spark DataFrame
spill.save_checkpoint_spark(spark_df, name="raw_spark_backup")

# List what you have saved
print(spill.list_checkpoints(storage="volume"))
# Output: ['raw_spark_backup', 'step_1_cleaned']
~~~
### ⚡Saving to Local Driver (Ephemeral)
Faster IO, but data is lost when the cluster restarts. Great for temporary caching during a notebook session.
~~~
# Save to /tmp on the driver
spill.save_checkpoint_pl(pl_df, name="temp_cache", storage="local")

# Load it back
df_cached = spill.load_checkpoint_pl(name="temp_cache", storage="local")
~~~
## 3. 💤Lazy Execution (Streaming)
If your data is too large to fit entirely in RAM, use Polars LazyFrames. VolumeSpiller supports this natively.
~~~
# 1. Convert Spark to Polars LazyFrame (eager=False)
# This scans the parquet files instead of reading them into memory immediately
lazy_pl = spill.spark_to_polars(spark_df, eager=False)

# 2. Build your query plan
q = lazy_pl.filter(pl.col("value") > 100).select("id", "value")

# 3. Execute or save
# You can sink directly back to a checkpoint without ever materializing in RAM
spill.save_checkpoint_pl(q, name="processed_lazy", storage="volume")
~~~
## 4. 🕝Automatic Timestamp Fixes
Spark uses Microseconds (us); Polars defaults to Nanoseconds (ns). Usually, this causes crashes when moving data. `VolumeSpiller` handles this silently.
~~~
# If your Polars DF has ns timestamps:
# pl_df = pl.DataFrame({"time": [datetime.now()]}) # default is ns

# This method detects 'ns' columns and casts them to 'ms' automatically
# before writing to Parquet, preventing Spark read errors.
spark_df = spill.polars_to_spark(pl_df)
~~~
## 5. 🛠️Dev vs. 🚀Prod Mode
The is_dev flag controls how the Volume is treated during initialization and teardown.
 - is_dev=True: Creates volume if missing. teardown() preserves files for debugging.
 - is_dev=False: Drops and recreates volume on init (clean slate). teardown() destroys the volume.
~~~
# PRODUCTION PATTERN .py
try:
    spill = VolumeSpiller(..., is_dev=False)
    
    # ... logic ...
    
finally:
    # ensuring the volume is dropped to save costs/storage
    spill.teardown()
~~~
OR
~~~
# PRODUCTION PATTERN .ipynb
# Cell 1
from databricks_scaffold import VolumeSpiller
import atexit

# Create function call at exit to remove volume no matter what
def cleanup_at_exit():
    """This function is the 'last man standing'."""
    try:
        if 'spiller' in globals() and not spill.is_dev:
            print("atexit: Cleaning up Unity Catalog Volume...")
            spill.teardown()
    except Exception as e:
        print(f"atexit cleanup failed: {e}")

atexit.register(cleanup_at_exit)

spill = VolumeSpiller(..., is_dev=False)

# Cell 2
# ... logic ...

# Cell 3
spill.teardown()
~~~
