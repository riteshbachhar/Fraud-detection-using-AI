"""
We recast the integer‑based columns following the logic rules outlined in the paper.
"Explainable Feature Engineering for Multi-class Money Laundering Classification"
This recasting is performed to optimize storage efficiency and reduce overall memory consumption."
Excluding "Sender_account" and "Receiver_account" variables.
"""

def recast(df):
    exclude = ['Sender_account', 'Receiver_account']

    for col in df.columns:
        if col not in exclude:
            dtype = df[col].dtype
            if dtype in (pl.Int64, pl.Int32):
              maxval = df[col].max()
              if maxval:
                  if maxval < 127:
                      df = df.with_columns(df[col].cast(pl.Int8).alias(col))
                  elif maxval < 32767:
                      df = df.with_columns(df[col].cast(pl.Int16).alias(col))
                  elif maxval < 2147483647:
                      df = df.with_columns(df[col].cast(pl.Int32).alias(col))

    return df