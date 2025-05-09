import os
import pandas as pd
import glob
import logging
from pathlib import Path

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/standardize_csv.log'),
        logging.StreamHandler()
    ]
)

def get_all_csv_files(base_dir):
    """Get all CSV files recursively from the base directory."""
    return glob.glob(os.path.join(base_dir, '**/*.csv'), recursive=True)

def get_common_columns(csv_files):
    """Get columns that are common across all CSV files."""
    if not csv_files:
        return []
    
    # Get columns from first file
    try:
        first_df = pd.read_csv(csv_files[0])
        common_columns = set(first_df.columns)
    except Exception as e:
        logging.error(f"Error reading first file {csv_files[0]}: {str(e)}")
        return []
    
    # Find intersection with columns from other files
    for file in csv_files[1:]:
        try:
            df = pd.read_csv(file)
            common_columns = common_columns.intersection(set(df.columns))
        except Exception as e:
            logging.error(f"Error reading {file}: {str(e)}")
            continue
    
    return sorted(list(common_columns))

def standardize_csv(file_path, common_columns):
    """Standardize a single CSV file with consistent column ordering."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Keep only common columns
        df = df[common_columns]
        
        # Ensure TIME is the first column
        cols = df.columns.tolist()
        if 'TIME' in cols:
            cols.remove('TIME')
            cols = ['TIME'] + sorted(cols)
            df = df[cols]
        
        # Save the standardized CSV
        df.to_csv(file_path, index=False)
        logging.info(f"Successfully standardized {file_path}")
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")

def main():
    # Get all CSV files from parent directory
    base_dir = 'NPPAD'  # Parent directory
    csv_files = get_all_csv_files(base_dir)
    logging.info(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        logging.error("No CSV files found! Please check if the files exist in the parent directory.")
        return
    
    # Get common columns
    common_columns = get_common_columns(csv_files)
    logging.info(f"Found {len(common_columns)} common columns across all files")
    logging.info(f"Common columns: {', '.join(common_columns)}")
    
    # Standardize each CSV file
    for file in csv_files:
        standardize_csv(file, common_columns)

if __name__ == "__main__":
    main() 