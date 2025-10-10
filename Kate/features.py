'''
Generate additional features
'''


def temporal_features(df):

  df['year'] = df['Date'].dt.year
  df['month'] = df['Date'].dt.month
  df['day_of_month'] = df['Date'].dt.day
  df['day_of_week'] = df['Date'].dt.dayofweek
  df['day_of_year'] = df['Date'].dt.dayofyear
  df['hour'] = df['Time'].dt.hour
  df['minute'] = df['Time'].dt.minute
  df['second'] = df['Time'].dt.second

  return df


def feature_engineer(df):

  # dropping column Laundering_type
  df = df.drop(columns = ["Laundering_type"])

  # fanin_30d: Number of unique sender accounts that sent money to a given receiver in the past 30 days.
  df['fanin_30d'] = df.groupby(['Receiver_account', pd.Grouper(key='Date', freq='30D')])['Sender_account'].transform('nunique')

  # fan_in_out_ratio: For each account, the number of unique inbound counterparties divided by the number of unique outbound counterparties in a 30-day window.
  df['fanout_30d'] = df.groupby(['Sender_account', pd.Grouper(key='Date', freq='30D')])['Receiver_account'].transform('nunique')
  df['fan_in_out_ratio'] = df['fanin_30d']/df['fanout_30d']
  df['fan_in_out_ratio'] = df['fan_in_out_ratio'].fillna(0)

  # fanin_intensity_ratio: Measures concentration—how many unique senders per daily inbound transaction.
  df['daily_receive'] = df.groupby(['Receiver_account', pd.Grouper(key='Date', freq='1D')])['Sender_account'].transform('nunique')
  df['fanin_intensity_ratio'] = df['fanin_30d']/df['daily_receive']
  df['fanin_intensity_ratio'] = df['fanin_intensity_ratio'].fillna(0)

  # amount_dispersion_std: Volatility of transaction amounts sent by each sender.
  df['amount_dispersion_std'] = df.groupby(['Sender_account'])['Amount'].transform('std')
  df['amount_dispersion_std'] = df['amount_dispersion_std'].fillna(0)

  # sent_to_received_ratio_monthly: For each account, total received amount divided by total sent amount over a monthly window.
  df['monthly_receive'] = df.groupby(['Receiver_account', pd.Grouper(key='Date', freq='ME')])['Amount'].transform('sum')
  df['monthly_send'] = df.groupby(['Sender_account', pd.Grouper(key='Date', freq='ME')])['Amount'].transform('sum')
  df['sent_to_received_ratio_monthly'] = df['monthly_receive']/df['monthly_send']
  df['sent_to_received_ratio_monthly'] = df['sent_to_received_ratio_monthly'].fillna(0)

  # back_and_forth_transfers: Number of unique transfers from a sender account to a receiver account in a single calendar day.
  df['back_and_forth_transfers'] = df.groupby(['Sender_account', 'Receiver_account', pd.Grouper(key='Date', freq='1D')])['Amount'].transform('count')

  # daily_receiver_transaction/weekly_receiver_transaction: Number of unique transaction from a receiver account in a single/week calendar day.
  df['daily_receiver_transaction'] = df.groupby(['Receiver_account', pd.Grouper(key='Date', freq='1D')])['Amount'].transform('count')
  df['weekly_receiver_transaction'] = df.groupby(['Receiver_account', pd.Grouper(key='Date', freq='1W')])['Amount'].transform('count')

  # daily_sender_transaction/weekly_sender_transaction: Number of unique transaction from a sender account in a single/week calendar day.
  df['daily_sender_transaction'] = df.groupby(['Sender_account', pd.Grouper(key='Date', freq='1D')])['Amount'].transform('count')
  df['weekly_sender_transaction'] = df.groupby(['Sender_account', pd.Grouper(key='Date', freq='1W')])['Amount'].transform('count')

  return df

def computing_circular_transaction(df):

  # circular_transaction_count: count of unique simple directed cycles that include an account (or a specific transaction edge) within 30 days.

  grouped = df.groupby(pd.Grouper(key='Date', freq='30D'))

  results = []

  for window_start, group in grouped:   # each 'group' is a DataFrame
      G = nx.DiGraph()
      # Now you can safely iterate rows of this group
      for _, row in group.iterrows():
          G.add_edge(row['Sender_account'], row['Receiver_account'])

      cycles = list(nx.simple_cycles(G))

      circular_count = collections.defaultdict(int)
      for cycle in cycles:
          for node in cycle:
              circular_count[node] += 1

      for node, count in circular_count.items():
          results.append({
              "Date": window_start,
              "Sender_account": node,
              "circular_transaction_count": count
          })

  cycle_features = pd.DataFrame(results)

  df = df.merge(
    cycle_features,
    on=['Date', 'Sender_account'],
    how='left'
    )

  df['circular_transaction_count'] = df['circular_transaction_count'].fillna(0)

  return df