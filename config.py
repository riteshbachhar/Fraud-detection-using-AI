"""
Project configuration and paths
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data/raw"
DATAPATH = DATA_DIR / "SAML-D.csv"
SAMPLE_DATAPATH = DATA_DIR / "sample_SAML-D.csv"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Kaggle dataset info
KAGGLE_DATASET = "berkanoztas/synthetic-transaction-monitoring-dataset-aml"

# Create directories if they don't exist
# DATA_DIR.mkdir(exist_ok=True)
# MODELS_DIR.mkdir(exist_ok=True)
# LOGS_DIR.mkdir(exist_ok=True)