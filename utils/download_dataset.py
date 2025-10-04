"""
Data related utilities for Kaggle datasets
"""

from pathlib import Path
# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

from config import DATA_DIR

def download_kaggle_dataset(
    dataset_name: str,
    output_dir: str = DATA_DIR,
    unzip: bool = True
):
    """
    Download dataset from Kaggle
    
    Parameters:
    -----------
    dataset_name : str
        Kaggle dataset identifier (owner/dataset-name)
    output_dir : str
        Directory where dataset will be saved
    unzip : bool
        Whether to unzip the downloaded file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize and authenticate
        api = KaggleApi()
        api.authenticate()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading to: {output_path.absolute()}")
        
        # Download dataset
        api.dataset_download_files(
            dataset_name,
            path=output_path,
            unzip=unzip
        )
        
        print("✓ Download complete")
        
        # List files
        files = sorted([f for f in output_path.glob("*") if f.is_file()])
        print(f"\nFiles ({len(files)}):")
        for file in files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name} ({size_mb:.2f} MB)")
        
        return True
        
    except ImportError:
        print("Error: Kaggle not installed")
        print("Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nSetup:")
        print("1. https://www.kaggle.com/settings/account")
        print("2. Create New API Token")
        print("3. Move kaggle.json to ~/.kaggle/")
        print("4. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)")
        return False


def download_saml_dataset(output_dir: str = DATA_DIR):
    """
    Download the SAML-D AML dataset
    
    Parameters:
    -----------
    output_dir : str
        Directory where dataset will be saved
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    return download_kaggle_dataset(
        dataset_name="berkanoztas/synthetic-transaction-monitoring-dataset-aml",
        output_dir=output_dir,
        unzip=True
    )