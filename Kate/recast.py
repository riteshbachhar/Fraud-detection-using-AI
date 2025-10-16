'''
We recast the integer‑based columns following the logic rules outlined in the paper.
"Explainable Feature Engineering for Multi-class Money Laundering Classification" 
This recasting is performed to optimize storage efficiency and reduce overall memory consumption."
Excluding "Sender_account" and "Receiver_account" variables.
''' 

def recast(df):
  exclude = ['Sender_account', 'Receiver_account']
  for col in df.select_dtypes(include=['int64', 'int32']).columns:
    if col not in exclude:
      if df[col].max() < 127:
        df[col] = df[col].astype('int8')
      elif df[col].max() < 32767:
        df[col] = df[col].astype('int16')
      elif df[col].max() < 2147483647:
        df[col] = df[col].astype('int32')
      else:
        df[col] = df[col].astype('int64')
  return df