def custom_split_polars(df: pl.DataFrame, validation_dt: int = 70, test_dt: int = 35):
    """
    Split a Polars DataFrame into train/validation/test by calendar-day cutoffs
    measured backwards from the dataset max Date.

    Parameters
    - df: polars.DataFrame with a datetime column named "Date" (string or datetime OK)
    - validation_dt: int days for validation window (e.g., 70)
    - test_dt: int days for test window (e.g., 35)

    Returns
    - train_df, validation_df, test_df  (all eager polars.DataFrame)
    """
    # ensure Date is a datetime type: try to get max, otherwise parse strings to Datetime
    try:
        max_date = df.select(pl.col("Date").max()).to_series()[0]
    except Exception:
        df = df.with_column(pl.col("Date").str.strptime(pl.Datetime, fmt=None).alias("Date"))
        max_date = df.select(pl.col("Date").max()).to_series()[0]

    test_cutoff = max_date - timedelta(days=test_dt)
    validation_cutoff = max_date - timedelta(days=validation_dt)

    test_set = df.filter(pl.col("Date") >= pl.lit(test_cutoff))
    validation_set = df.filter(
        (pl.col("Date") >= pl.lit(validation_cutoff)) & (pl.col("Date") < pl.lit(test_cutoff))
    )
    train_set = df.filter(pl.col("Date") < pl.lit(validation_cutoff))

    return train_set, validation_set, test_set