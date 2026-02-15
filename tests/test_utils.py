def test_clean_column_names(spark):
    from databricks_scaffold.utils import clean_column_names
    data = [(1, "foo")]
    # Col names with spaces, dashes, and dots
    df = spark.createDataFrame(data, ["ID #", "First.Name-Extra"])
    
    clean_df = clean_column_names(df)
    assert clean_df.columns == ["ID", "First_Name_Extra"]

def test_keep_duplicates(spark):
    from databricks_scaffold.utils import keep_duplicates
    data = [(1, "A"), (2, "B"), (1, "A"), (3, "C")]
    df = spark.createDataFrame(data, ["id", "val"])
    
    dupes = keep_duplicates(df, subset=["id", "val"])
    assert dupes.count() == 2
    assert all(row["id"] == 1 for row in dupes.collect())