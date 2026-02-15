# databricks-scaffold
Productivity booster functions for using polars on Databricks. Ease the pain of working with small data on Databricks. Single node rebellion 🔥

Helps the process of switching back and forth between PySpark Dataframe and polars Dataframe and _more_.

Features:
- Volumespill: 
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

# Usage
Install it:
~~~
%pip install git+https://github.com/spookytomtom/databricks-scaffold.git
~~~

## Quickstart
~~~
from databricks_scaffold import VolumeSpiller

spill = VolumeSpiller(
    catalog="main", 
    schema="default", 
    volume_name="temp_spill", 
    is_dev=True
)

# Convert Spark to Polars (spills to volume transparently)
pl_df = spill.spark_to_polars(spark_df)

# Do Polars work...
pl_df = pl_df.group_by("col").count()

# Convert back to Spark
final_spark_df = spill.polars_to_spark(pl_df)
~~~

## In a .py
Avoid leaving any volume behind using `try` `finally`
~~~
from databricks_scaffold import VolumeSpiller

try:
    spill = VolumeSpiller(
        catalog="main", 
        schema="default", 
        volume_name="temp_spill", 
        is_dev=True
    )

    # Convert Spark to Polars (spills to volume transparently)
    pl_df = spill.spark_to_polars(spark_df)

    # Do Polars work...
    pl_df = pl_df.group_by("col").count()

    # Convert back to Spark
    final_spark_df = spill.polars_to_spark(pl_df)

finally:
    spill.teardown()
~~~
## In a .ipynb
~~~
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

spill = VolumeSpiller(
    catalog="main", 
    schema="default", 
    volume_name="temp_spill", 
    is_dev=True
)

# Cell 1
# Convert Spark to Polars (spills to volume transparently)
pl_df = spill.spark_to_polars(spark_df)

# Cell 2
# Do Polars work...
pl_df = pl_df.group_by("col").count()

# Cell 3
# Convert back to Spark
final_spark_df = spill.polars_to_spark(pl_df)

# Cell 4
spill.teardown()
~~~

