#!/usr/bin/env python3
"""
Data preparation for temporal graph model.
Create graph snapshots from transaction data.
"""

import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data
import yaml
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'utils'))
from config import DATAPATH
try:
    from utils import load_config  # Works for scripts in src/
except ImportError:
    from src.utils import load_config  # Works for notebooks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Data Processor Class
class TemporalGraphDataProcessor:
    """
    Processes transaction data to create temporal graph snapshots.
    """

    def __init__(self, time_window):
        self.time_window = time_window

    def load_and_preprocess(self, filepath=None):
        """Load SAML-D dataset and perform initial preprocessing"""
        if filepath is None:
            if DATAPATH.exists():
                df = pd.read_csv(DATAPATH)
            else:
                logger.error("Data path not found in config.py")
                raise FileNotFoundError(f"Required data file not found: {DATAPATH}")

        logger.info("Loading and preprocessing data...")

        # Combine date and time into datetime
        df.insert(0, 'datetime', pd.to_datetime(df["Date"] + ' ' + df["Time"], format='%Y-%m-%d %H:%M:%S'))
        df.drop(columns=['Date', 'Time'], inplace=True)

        logger.info(f"Loaded {len(df)} transactions")
        logger.info(f"Suspicious transactions: {df['Is_laundering'].sum()} ({df['Is_laundering'].mean()*100:.3f}%)")

        return df

    def calculate_account_features(self, df):
        """ Calculate account-level features over the specified time window. """
        logger.info(f"Calculating account-level features over time window: {self.time_window}")

        start_date = df['datetime'].min().date()
        end_date = (df['datetime'].max() + pd.Timedelta(days=1)).date() 

        # DataFrame to store account stats
        account_stats = pd.DataFrame({})

        for window_start in pd.date_range(start=start_date, end=end_date, freq=self.time_window):
            window_end = window_start + pd.Timedelta(self.time_window)
            window_txns = df[(df['datetime'] >= pd.Timestamp(window_start)) & (df['datetime'] < pd.Timestamp(window_end))]
            active_accounts = list(set(window_txns['Sender_account'].unique()).union(set(window_txns['Receiver_account'].unique())))

            logger.info(f"Processing window {window_start.date()} to {(window_end - pd.Timedelta('1D')).date()}")

            # Sent and received transaction stats
            sent_txns = window_txns.groupby(['Sender_account']).agg({
                        'Receiver_account': ['size', 'nunique'],
                        'Amount': ['median', 'std', 'sum']
                        }).reset_index()
            sent_txns.columns = ['account', 'sent_txns_count', 'fan_out', 'med_sent_amt', 'std_sent_amt', 'total_sent_amt']
            received_txns = window_txns.groupby(['Receiver_account']).agg({
                        'Sender_account': ['size', 'nunique'],
                        'Amount': ['median', 'std', 'sum']
                        }).reset_index()
            received_txns.columns = ['account', 'recv_txns_count', 'fan_in', 'med_recv_amt', 'std_recv_amt', 'total_recv_amt']

            # Max transaction between accounts
            sent_recv_txns = window_txns.groupby(['Sender_account', 
                                            'Receiver_account']).agg({
                                            'Amount': ['count', 'sum']
                                            }).reset_index()
            sent_recv_txns.columns = ['Sender_account', 'Receiver_account', 'sent_txn_count', 'sent_txn_amount']
            max_sent_txns = sent_recv_txns.groupby('Sender_account').agg({
                'sent_txn_count': 'max',
                'sent_txn_amount': 'max'
            }).reset_index()
            max_sent_txns.columns = ['account', 'max_sent_txn_count', 'max_sent_txn_amt']

            recv_sent_txns = window_txns.groupby(['Receiver_account', 
                                 'Sender_account']).agg({
                                     'Amount': ['count', 'sum']
                                 }).reset_index()
            recv_sent_txns.columns = ['Receiver_account', 'Sender_account', 'recv_txn_count', 'recv_txn_amount']
            max_recv_txns = recv_sent_txns.groupby('Receiver_account').agg({
                'recv_txn_count': 'max',
                'recv_txn_amount': 'max'
            }).reset_index()
            max_recv_txns.columns = ['account', 'max_recv_txn_count', 'max_recv_txn_amt']

            # Initialize window dataframe
            window_data = pd.DataFrame({
                'window_start': window_start,
                'account': active_accounts})
            window_data = window_data.merge(sent_txns, on='account', how='left') # sent_txns_count
            window_data = window_data.merge(received_txns, on='account', how='left') # received_txns_count
            window_data = window_data.merge(max_sent_txns, on='account', how='left')
            window_data = window_data.merge(max_recv_txns, on='account', how='left')
            window_data.fillna(0, inplace=True)

            # Calculated
            window_data['sent_recv_ratio'] = window_data.apply(
                lambda r: r['sent_txns_count'] / r['recv_txns_count'] if r['recv_txns_count'] > 0 else -1, axis=1
            )

            window_data['fanout_fanin_ratio'] = window_data.apply(
                lambda r: r['fan_out'] / r['fan_in'] if r['fan_in'] > 0 else -1, axis=1
            )

            window_data['total_txns_amt'] = window_data['total_sent_amt'] + window_data['total_recv_amt']
            del window_data['total_sent_amt']
            del window_data['total_recv_amt']


            # Append to main dataframe
            account_stats = pd.concat([account_stats, window_data], ignore_index=True)
        
        # Set data types
        account_stats = account_stats.astype({
            'sent_txns_count': 'int32',
            'recv_txns_count': 'int32',
            'fan_out': 'int32',
            'fan_in': 'int32',
            'max_sent_txn_count': 'int32',
            'max_recv_txn_count': 'int32',
            'sent_recv_ratio': 'float32',
            'fanout_fanin_ratio': 'float32'
        })

        # Convert to log1p
        columns = ['med_sent_amt', 'std_sent_amt', 'med_recv_amt', 'std_recv_amt', 
           'max_sent_txn_amt', 'max_recv_txn_amt', 'total_txns_amt']

        for col in columns:
            account_stats['log_' + col] = np.log1p(account_stats[col]).astype('float32')
        account_stats = account_stats.drop(columns, axis=1)

        logger.info("Account-level features calculation completed.")
        return account_stats

    def engineer_features(self, df):
        """Engineer transaction-level features."""
        logging.info("Engineering transaction features...")
        
        # Time-based features (more granular)
        df['hour'] = df['datetime'].dt.hour.astype('int8')
        df['month'] = df['datetime'].dt.month.astype('int8')
        df['day_of_week'] = df['datetime'].dt.dayofweek.astype('int8')
        df['day_of_month'] = df['datetime'].dt.day.astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype('int8')  # Night transactions
        
        # Amount-based features
        df['log_amount'] = np.log1p(df['Amount']).astype('float32')
        
        # Geographic risk features
        df['cross_border'] = (df['Payment_type'] == 'Cross-border').astype('int8')
        risky_countries = {'Mexico', 'Turkey', 'Morocco', 'UAE'}
        df['high_risk_sender'] = df['Sender_bank_location'].isin(risky_countries).astype('int8')
        df['high_risk_receiver'] = df['Receiver_bank_location'].isin(risky_countries).astype('int8')
        
        # Currency features
        df['currency_mismatch'] = (df['Payment_currency'] != df['Received_currency']).astype('int8')
        
        # Convert target
        df['Is_laundering'] = df['Is_laundering'].astype('int8')
        
        # Clean up
        columns_to_drop = ['Amount', 'Sender_bank_location', 'Receiver_bank_location', 
                           'Payment_currency', 'Received_currency', 'Laundering_type']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Lebel encoding
        # categorical_cols = ['Payment_currency', 'Received_currency', 'Sender_bank_location', 
        #                    'Receiver_bank_location', 'Payment_type']
        categorical_cols = ['Payment_type']
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            # Not saving it for now
        df = df.drop(categorical_cols, axis=1)
        
        return df

    def create_temporal_snapshots(self, df, df_account_stats):
        """
        Create temporal graph snapshots from transaction data and account features.
        """
        logging.info("Creating temporal graph snapshots...")
        
        # Global account mapping
        all_accounts = list(set(df['Sender_account'].unique()) | set(df['Receiver_account'].unique()))
        global_account_to_idx = {acc: idx for idx, acc in enumerate(all_accounts)}
        global_num_nodes = len(all_accounts)
        
        # Store graph data
        snapshots = []

        for window_start in sorted(df_account_stats['window_start'].unique()):
            window_start = pd.to_datetime(window_start)
            window_end = window_start + pd.Timedelta(self.time_window)
            window_end_print = (window_end - pd.Timedelta('1D')).strftime('%Y-%m-%d') # One day less for printing
            window_start_str = window_start.strftime('%Y-%m-%d')
            window_end_str = window_end.strftime('%Y-%m-%d')
            logging.info(f"Snapshot window: {window_start_str} to {window_end_print}")
            
            # Get transactions in current window
            window_mask = (df['datetime'] >= window_start_str) & (df['datetime'] < window_end_str)
            window_trnx_data = df[window_mask].copy()

            # Account features for this window
            window_account_stats = df_account_stats[df_account_stats['window_start'] == window_start_str].copy()
            
            if len(window_trnx_data) > 0:
                graph_data = self._create_graph_snapshot(
                    window_trnx_data, window_account_stats,
                    window_start_str, global_account_to_idx, global_num_nodes
                )
                if graph_data is not None:
                    snapshots.append(graph_data)

        logging.info(f"Created {len(snapshots)} temporal snapshots")
        return snapshots, global_num_nodes
    
    def _create_graph_snapshot(self, window_trnx_data, window_accounts_features, 
                              timestamp, global_account_to_idx, global_num_nodes):
        """Create enhanced graph snapshot"""
        if len(window_trnx_data) == 0:
            return None

        # Enhanced edge features
        edge_feature_columns = [
            'Payment_type_encoded', 'log_amount', 'month', 'day_of_week', 'hour', 
            'currency_mismatch', 'cross_border', 'high_risk_sender', 'high_risk_receiver',
        ]
        
        # Filter available columns
        edge_feature_columns = [col for col in edge_feature_columns if col in window_trnx_data.columns]

        # Node features
        node_feature_columns = ['sent_txns_count', 'fan_out', 'recv_txns_count', 'fan_in', 
                               'max_sent_txn_count', 'max_recv_txn_count', 'sent_recv_ratio', 
                               'fanout_fanin_ratio', 'log_med_sent_amt', 'log_std_sent_amt', 
                               'log_med_recv_amt', 'log_std_recv_amt', 'log_max_sent_txn_amt', 
                               'log_max_recv_txn_amt', 'log_total_txns_amt']

        # Create mappings and features
        sender_mapped = window_trnx_data['Sender_account'].map(global_account_to_idx)
        receiver_mapped = window_trnx_data['Receiver_account'].map(global_account_to_idx)
        edge_index = np.column_stack((sender_mapped, receiver_mapped))
        edge_features = window_trnx_data[edge_feature_columns].values
        transaction_labels = window_trnx_data['Is_laundering'].values

        # Node features
        node_features = np.zeros((global_num_nodes, len(node_feature_columns)))
        try:
            window_accounts_features['global_idx'] = window_accounts_features['account'].map(global_account_to_idx)
            node_features[window_accounts_features['global_idx'].values] = window_accounts_features[node_feature_columns].values
        except: 
            raise ValueError("Error in mapping account features to global indices.")

        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        transaction_labels = torch.tensor(transaction_labels, dtype=torch.float)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=transaction_labels,
            timestamp=timestamp,
            num_nodes=global_num_nodes
        )

def main():
    """Main preprocessing pipeline"""
    # Load config
    script_dir = Path(__file__).parent
    config = load_config(script_dir.parent / 'config.yaml')
    logger.info("Starting data preprocessing...")
    
    # Create output directories
    Path('data').mkdir(parents=True, exist_ok=True)

    try:
        processor = TemporalGraphDataProcessor(config['preprocessing']['time_window'])      # Initialize temporal graph processor
        df = processor.load_and_preprocess()                                                # Load and simple preprocessing
        df_account_stats = processor.calculate_account_features(df)       # Calculate account-level features over a time window
        df = processor.engineer_features(df)                              # Engineer transaction-level features
        snapshots, global_num_nodes = processor.create_temporal_snapshots(df, df_account_stats)       # Create temporal graph snapshots 

        # Save processed data
        torch.save(snapshots, 'data/temporal_graph_snapshots.pth')
        torch.save({'num_nodes': global_num_nodes}, 'data/graph_info.pth')

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()