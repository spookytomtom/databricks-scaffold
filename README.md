# databricks-scaffold
Productivity booster functions for using polars on Databricks. Ease the pain of working with small data on Databricks. Single node rebellion 🔥

Helps the process of switching back and forth between PySpark Dataframe and polars Dataframe and _more_.

Features:
- VolumeSpiller: 
  - Utilize Unity Catalog Volume as a spillage bucket making the switch between pyspark and polars faster or just to save checkpoint parquet files.
  - Utilize the driver node file system as a temporary scan / sink storage for polars.
- adds utility functions missing from pyspark, such as 
  - glimpse(), 
  - keep_duplicates(), 
  - shape(), 
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