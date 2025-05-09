import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/classification_feature_engineering_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class FeatureEngineer:
    def __init__(self, window_size=5): #50 sec
        """
        Initialize feature engineering with window size for time-based features.
        window_size: number of time steps to consider for statistical features
        """
        self.window_size = window_size
        self.scaler = None
        
    def calculate_statistical_features(self, df):
        """Calculate statistical features for each window."""
        # Exclude time column
        df = df.drop('TIME', axis=1)
        
        features = []
        
        # Basic statistics
        features.extend([
            df.rolling(window=self.window_size, min_periods=1).mean(),
            df.rolling(window=self.window_size, min_periods=1).std(),
            df.rolling(window=self.window_size, min_periods=1).min(),
            df.rolling(window=self.window_size, min_periods=1).max()
        ])
        
        # Rate of change
        features.append(df.diff().fillna(0))
        
        # Second derivative (acceleration)
        features.append(df.diff().diff().fillna(0))
        
        # Combine all features
        feature_df = pd.concat(features, axis=1)
        
        # Rename columns to indicate feature type
        new_columns = []
        for col in df.columns:
            new_columns.extend([
                f'{col}_mean',
                f'{col}_std',
                f'{col}_min',
                f'{col}_max',
                f'{col}_rate_of_change',
                f'{col}_acceleration'
            ])
        feature_df.columns = new_columns
        
        # Fill any remaining NaN values with 0
        feature_df = feature_df.fillna(0)
        
        return feature_df
    
    def normalize_features(self, df, fit=False):
        """Normalize features using MinMaxScaler."""
        # Check for any remaining NaN values
        if df.isna().any().any():
            logging.warning("NaN values found in features before normalization. Filling with 0.")
            df = df.fillna(0)
            
        if self.scaler is None or fit:
            self.scaler = MinMaxScaler()
            normalized_data = self.scaler.fit_transform(df)
        else:
            normalized_data = self.scaler.transform(df)
        
        return pd.DataFrame(normalized_data, columns=df.columns)
    
    def process_simulation(self, csv_path, output_dir):
        """Process a single simulation file."""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Calculate features
            statistical_features = self.calculate_statistical_features(df)
            
            all_features = statistical_features
            
            # Normalize features
            normalized_features = self.normalize_features(all_features, fit=True)
            
            # Save processed data
            output_path = os.path.join(output_dir, os.path.basename(csv_path))
            normalized_features.to_csv(output_path, index=False)
            
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
        
        logging.info("Feature engineering completed!")

def main():
    """Main function to run feature engineering."""
    # Input and output directories
    input_dir = "NPPAD_for_classifiers"
    output_dir = "NPPAD_for_classifiers_features"
    
    # Initialize feature engineer
    engineer = FeatureEngineer(window_size=30)
    
    # Process dataset
    engineer.process_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main() 