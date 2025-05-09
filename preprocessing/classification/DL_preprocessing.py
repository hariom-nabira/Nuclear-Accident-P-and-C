import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from tqdm import tqdm
import joblib

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/dl_preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class DLPreprocessor:
    def __init__(self, sequence_length=100):  # 1000 seconds (100 * 10s intervals)
        """
        Initialize DL preprocessor with sequence length.
        sequence_length: number of time steps to consider for each sequence
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_sequence(self, df):
        """Prepare a sequence of fixed length from the time series data."""
        # Exclude time column
        df = df.drop('TIME', axis=1)
        
        # If sequence is longer than required, take the last sequence_length rows
        if len(df) > self.sequence_length:
            df = df.iloc[-self.sequence_length:]
        # If sequence is shorter, pad with the first row
        elif len(df) < self.sequence_length:
            padding = pd.DataFrame([df.iloc[0]] * (self.sequence_length - len(df)))
            df = pd.concat([padding, df], ignore_index=True)
            
        return df
    
    def standardize_features(self, df, fit=False):
        """Standardize features using StandardScaler."""
        # Check for any remaining NaN values
        if df.isna().any().any():
            logging.warning("NaN values found in features before standardization. Filling with 0.")
            df = df.fillna(0)
            
        if fit:
            standardized_data = self.scaler.fit_transform(df)
            self.feature_names = df.columns
        else:
            standardized_data = self.scaler.transform(df)
        
        return pd.DataFrame(standardized_data, columns=df.columns)
    
    def process_simulation(self, csv_path, output_dir):
        """Process a single simulation file."""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Prepare sequence
            sequence = self.prepare_sequence(df)
            
            # Standardize features
            standardized_sequence = self.standardize_features(sequence, fit=True)
            
            # Save processed data
            output_path = os.path.join(output_dir, os.path.basename(csv_path))
            standardized_sequence.to_csv(output_path, index=False)
            
            logging.info(f"Successfully processed {csv_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {csv_path}: {str(e)}")
            return False
    
    def process_dataset(self, input_dir, output_dir):
        """Process all simulations in the dataset."""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process each accident type
        for accident_type in os.listdir(input_dir):
            accident_dir = os.path.join(input_dir, accident_type)
            if not os.path.isdir(accident_dir):
                continue
            
            logging.info(f"Processing accident type: {accident_type}")
            
            # Create corresponding directory in output
            output_accident_dir = os.path.join(output_dir, accident_type)
            os.makedirs(output_accident_dir, exist_ok=True)
            
            # Process each simulation
            for csv_file in tqdm(os.listdir(accident_dir), desc=f"Processing {accident_type}"):
                if not csv_file.endswith('.csv'):
                    continue
                
                csv_path = os.path.join(accident_dir, csv_file)
                self.process_simulation(csv_path, output_accident_dir)
        
        # Save scaler for later use
        scaler_path = os.path.join(output_dir, "standard_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        logging.info(f"Saved standard scaler to {scaler_path}")
        
        logging.info("DL preprocessing completed!")

def main():
    """Main function to run DL preprocessing."""
    # Input and output directories
    input_dir = "NPPAD_for_classifiers"
    output_dir = "NPPAD_for_classifiers_dl"
    
    # Initialize preprocessor
    preprocessor = DLPreprocessor(sequence_length=100)
    
    # Process dataset
    preprocessor.process_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main() 