import polars as pl
from typing import List, Dict


def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build and return engineered features from a Polars DataFrame.
    """

    # Convert datetime
    df = df.with_columns(
    pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d").alias("Date"),
    pl.col("Time").str.strptime(pl.Time, "%H:%M:%S").alias("Time"))

    # Dropping column Laundering_type
    df = df.drop("Laundering_type")

    # Temporal features
    df = temporal_features(df)

    # Risk features
    df = risk_features(df)

    # Rolling window computing
    specs = [
    {"name":"fanin_30d", "kind":"rolling", "type":"fanin", "period_days":30, "every":"1d"},
    {"name":"fanout_30d", "kind":"rolling", "type":"fanout", "period_days":30, "every":"1d"},
    {"name":"daily_recieve", "kind":"rolling", "type":"fanin", "period_days":1, "every":"1d"},
    {"name":"monthly_receive", "kind":"monthly", "side":"receive"},
    {"name":"monthly_send",    "kind":"monthly", "side":"send"},
    {"name":"back_and_forth_transfers", "kind":"daily_pair_count"},
    ]
    lazy_with_features = build_window_features_lazy(df, specs, amount_col="Amount", label_choice="left")
    plan = (
    lazy_with_features
    .sort(["Sender_account", "Date"])
    .with_columns([pl.col("Sender_account").set_sorted(), pl.col("Date").set_sorted()])
    )
    df_streamed = plan.collect(engine="streaming")
    df = df_streamed.sort("__row_idx").drop("__row_idx")

    # More computation
    lazy_with_derived = compute_derived_features_lazy(lazy_with_features)

    # Before streaming collect: pick a primary grouping ordering that matches your rolling computations.
    # If most rolling features used Receiver_account then Date, use that; otherwise use the grouping you chose.
    plan_derived = (
        lazy_with_derived
        .sort(["Sender_account", "Date"])
        .with_columns([pl.col("Sender_account").set_sorted(), pl.col("Date").set_sorted()])
    )

    df_streamed = plan_derived.collect(engine="streaming")
    df = df_streamed.sort("__row_idx").drop("__row_idx")

    # Daily & Weekly transaction count
    result_lazy = add_daily_weekly_transaction_counts(df.lazy())
    df = result_lazy.collect(engine="streaming")

    return df


# Temporal features
def temporal_features(df):

    return df.with_columns([
        df["Date"].dt.year().alias("year"),
        df["Date"].dt.month().alias("month"),
        df["Date"].dt.day().alias("day_of_month"),
        df["Date"].dt.weekday().alias("day_of_week"),
        df["Date"].dt.ordinal_day().alias("day_of_year"),
        df["Time"].dt.hour().alias("hour"),
        df["Time"].dt.minute().alias("minute"),
        df["Time"].dt.second().alias("second"),
    ])

# Risk features
high_risk_countries = ['Mexico', 'Turkey', 'Morocco', 'UAE']

def risk_features(df):

    return df.with_columns([
        (df["Payment_currency"] != df["Received_currency"]).cast(pl.Int8).alias("currency_mismatch"),
        (df["Payment_type"] == "Cross-border").cast(pl.Int8).alias("cross_border"),
        df["Sender_bank_location"].is_in(high_risk_countries).cast(pl.Int8).alias("high_risk_sender"),
        df["Receiver_bank_location"].is_in(high_risk_countries).cast(pl.Int8).alias("high_risk_receiver")])

def build_window_features_lazy(
    df,
    specs,
    date_col="Date",
    sender_col="Sender_account",
    receiver_col="Receiver_account",
    amount_col="Amount",
    index_name="__row_idx",
    label_choice="left",
):
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    lf = lf.with_columns(pl.arange(0, pl.len()).over(pl.lit(True)).alias(index_name))
    out_lf = lf

    for spec in specs:
        kind = spec.get("kind", "rolling")

        if kind == "rolling":
            # existing rolling logic (no change)
            name = spec["name"]
            direction = spec["type"]  # "fanin" or "fanout"
            period_days = int(spec["period_days"])
            every = spec.get("every", "1d")

            if direction == "fanin":
                group_by = receiver_col
                agg_on = sender_col
            else:
                group_by = sender_col
                agg_on = receiver_col

            win_label = label_choice
            strategy = "forward" if win_label == "left" else "backward"

            right = (
                lf
                .sort([group_by, date_col])
                .group_by_dynamic(
                    index_column=date_col,
                    every=every,
                    period=f"{period_days}d",
                    group_by=group_by,
                    closed="both",
                    label=win_label
                )
                .agg(pl.col(agg_on).n_unique().alias(name))
                .sort([group_by, date_col])
            )

            left = out_lf.sort([group_by, date_col])

            out_lf = left.join_asof(
                right,
                left_on=date_col,
                right_on=date_col,
                by=group_by,
                strategy=strategy,
            )

        elif kind == "monthly":
            # existing monthly logic (no change)
            name = spec["name"]
            side = spec.get("side", "receive")
            group_col = receiver_col if side == "receive" else sender_col

            monthly_agg = (
                lf
                .with_columns(pl.col(date_col).dt.truncate("1mo").alias("__month"))
                .group_by([group_col, "__month"])
                .agg(pl.col(amount_col).sum().alias(name))
            )

            out_lf = (
                out_lf
                .with_columns(pl.col(date_col).dt.truncate("1mo").alias("__month"))
                .join(monthly_agg, on=[group_col, "__month"], how="left")
                .drop("__month")
            )

        elif kind == "daily_pair_count":
            # NEW: back_and_forth_transfers (exact-match on day + pair)
            name = spec["name"]  # e.g., "back_and_forth_transfers"
            # day key = calendar day (truncate to 1 day)
            day_key = "__day"
            # compute counts per sender/receiver/day using lf (lazy)
            pair_daily_agg = (
                lf
                .with_columns(pl.col(date_col).dt.truncate("1d").alias(day_key))
                .group_by([sender_col, receiver_col, day_key])
                .agg(pl.len().alias(name))  # .len() counts rows in group
            )

            # attach day key to working frame and join exact on pair + day
            out_lf = (
                out_lf
                .with_columns(pl.col(date_col).dt.truncate("1d").alias(day_key))
                .join(pair_daily_agg, on=[sender_col, receiver_col, day_key], how="left")
                .fill_null(0)       # optional: replace nulls with 0
                .with_columns(pl.col(name).cast(pl.Int64))  # ensure integer type
                .drop(day_key)
            )

        else:
            raise ValueError("spec kind must be 'rolling', 'monthly', or 'daily_pair_count'")

    return out_lf


def compute_derived_features_lazy(
    lf: pl.LazyFrame,
    *,
    fanin_col: str = "fanin_30d",
    fanout_col: str = "fanout_30d",
    daily_receive_col: str = "daily_receive",
    monthly_receive_col: str = "monthly_receive",
    monthly_send_col: str = "monthly_send",
    amount_col: str = "Amount",
    sender_col: str = "Sender_account",
    receiver_col: str = "Receiver_account",
    index_name: str = "__row_idx",
) -> pl.LazyFrame:
    """
    Take a LazyFrame and return a LazyFrame with derived features:
      - fan_in_out_ratio (safe division, 0 when denom missing or zero)
      - fanin_intensity_ratio (fanin_30d / daily_receive, 0 when denom missing or zero)
      - amount_dispersion_std (per-sender std of Amount, filled 0 when null)
      - sent_to_received_ratio_monthly (monthly_receive / monthly_send, 0 when denom missing or zero)

    If `daily_receive` is not present in lf.schema(), it is computed lazily as the
    per-receiver unique-senders per calendar day (dt.truncate("1d")) and joined back.
    The function is fully lazy; call .collect(...) when ready.
    """
    # ensure lazy input
    lf = lf if isinstance(lf, pl.LazyFrame) else lf.lazy()

    # Attempt to read schema; if unavailable assume missing and compute
    try:
        schema = lf.schema()
        has_daily = daily_receive_col in schema
    except Exception:
        has_daily = False

    # If daily_receive missing, compute it lazily (exact day bucket of unique senders per receiver)
    if not has_daily:
        day_key = "__day_for_daily_receive"
        daily_receive_agg = (
            lf
            .with_columns(pl.col("Date").dt.truncate("1d").alias(day_key))
            .group_by([receiver_col, day_key])
            .agg(pl.col(sender_col).n_unique().alias(daily_receive_col))
        )
        lf = (
            lf
            .with_columns(pl.col("Date").dt.truncate("1d").alias(day_key))
            .join(daily_receive_agg, on=[receiver_col, day_key], how="left")
            .drop(day_key)
        )

    # safe division helper expression
    def safe_div_expr(num: str, den: str, out_name: str):
        return (
            pl.when(pl.col(den).is_null() | (pl.col(den) == 0))
              .then(0.0)
              .otherwise(pl.col(num).cast(pl.Float64) / pl.col(den).cast(pl.Float64))
              .alias(out_name)
        )

    fan_in_out_expr = safe_div_expr(fanin_col, fanout_col, "fan_in_out_ratio")
    fanin_intensity_expr = safe_div_expr(fanin_col, daily_receive_col, "fanin_intensity_ratio")
    sent_to_received_monthly_expr = safe_div_expr(monthly_receive_col, monthly_send_col, "sent_to_received_ratio_monthly")

    # per-sender std aggregation (lazy) and join back
    sender_std_agg = (
        lf
        .select([sender_col, amount_col])
        .group_by(sender_col)
        .agg(pl.col(amount_col).std().alias("__amount_std"))
    )

    out = (
        lf
        .join(sender_std_agg, on=sender_col, how="left")
        .with_columns(
            pl.col("__amount_std").cast(pl.Float64).fill_null(0.0).alias("amount_dispersion_std")
        )
        .drop("__amount_std")
        .with_columns([
            fan_in_out_expr,
            fanin_intensity_expr,
            sent_to_received_monthly_expr
        ])
    )

    return out


def add_daily_weekly_transaction_counts(
    lf: pl.LazyFrame,
    date_col: str = "Date",
    sender_col: str = "Sender_account",
    receiver_col: str = "Receiver_account",
    amount_col: str = "Amount",
) -> pl.LazyFrame:
    """
    Return a LazyFrame with four new columns:
      - daily_receiver_transaction
      - weekly_receiver_transaction
      - daily_sender_transaction
      - weekly_sender_transaction

    The function keeps the pipeline lazy: it returns a LazyFrame you can .collect() later.
    """
    # base lazy frame with truncated calendar buckets
    base = lf.with_columns(
        [
            pl.col(date_col).dt.truncate("1d").alias("_day"),
            pl.col(date_col).dt.truncate("1w").alias("_week"),
        ]
    )

    # aggregations (still lazy)
    agg_daily_recv = (
        base
        .group_by([receiver_col, "_day"])
        .agg(pl.count(amount_col).alias("daily_receiver_transaction"))
    )

    agg_weekly_recv = (
        base
        .group_by([receiver_col, "_week"])
        .agg(pl.count(amount_col).alias("weekly_receiver_transaction"))
    )

    agg_daily_sndr = (
        base
        .group_by([sender_col, "_day"])
        .agg(pl.count(amount_col).alias("daily_sender_transaction"))
    )

    agg_weekly_sndr = (
        base
        .group_by([sender_col, "_week"])
        .agg(pl.count(amount_col).alias("weekly_sender_transaction"))
    )

    # join aggregated counts back to the base lazyframe
    out = (
        base
        .join(agg_daily_recv, left_on=[receiver_col, "_day"], right_on=[receiver_col, "_day"], how="left")
        .join(agg_weekly_recv, left_on=[receiver_col, "_week"], right_on=[receiver_col, "_week"], how="left")
        .join(agg_daily_sndr, left_on=[sender_col, "_day"], right_on=[sender_col, "_day"], how="left")
        .join(agg_weekly_sndr, left_on=[sender_col, "_week"], right_on=[sender_col, "_week"], how="left")
        .with_columns(
            [
                pl.coalesce(["daily_receiver_transaction", pl.lit(0)]).cast(pl.Int64),
                pl.coalesce(["weekly_receiver_transaction", pl.lit(0)]).cast(pl.Int64),
                pl.coalesce(["daily_sender_transaction", pl.lit(0)]).cast(pl.Int64),
                pl.coalesce(["weekly_sender_transaction", pl.lit(0)]).cast(pl.Int64),
            ]
        )
        .drop(["_day", "_week"])
    )

    return out
