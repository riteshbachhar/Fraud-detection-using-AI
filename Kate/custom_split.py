'''
Custom split in terms of time, with default cutoffs of 70 days and 35 days for the validation and test sets, respectively.
Inputs:
- df: DataFrame
- validation_dt: default is the last 70 days
- test_dt: default is the last 35 days”
Outputs:
- train_set
- validation_set
- test_set
'''


def custom_split(df, validation_dt=70, test_dt=35):
  # validation_cut_off=70: selects transactions for validation set, from the final 70 days prior to the dataset's latest date
  # test_cut_off=40: selects transactions for test set, from the final 70 days prior to the dataset's latest date

  test_cutoff = df['Date'].max() - pd.Timedelta(days=test_dt)
  validation_cutoff = df['Date'].max() - pd.Timedelta(days=validation_dt)

  test_set = df[df.Date >= test_cutoff]
  validation_set = df[(df.Date >= validation_cutoff) & (df.Date < test_cutoff)]
  train_set = df[df.Date < validation_cutoff]

  return train_set, validation_set, test_set