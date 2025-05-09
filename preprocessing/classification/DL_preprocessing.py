import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from tqdm import tqdm

# Set up logging
log_dir = os.path.join('logs', 'preprocessing', 'classification', 'DL')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'dl_preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class DLPreprocessor:
    def __init__(self, window_size=5): #50 sec
        """
        Initialize the DL preprocessor
        
        Args:
            window_size (int): Size of the time window for feature engineering
        """
        self.window_size = window_size
        self.scaler = StandardScaler()
        logging.info(f"Initialized DL preprocessor with window size: {window_size}")

    def _calculate_statistical_features(self, data):
        """
        Calculate statistical features for a time window
        
        Args:
            data (np.array): Time series data for a window
            
        Returns:
            np.array: Statistical features
        """
        data = data.drop('TIME', axis=1)
        features = []
        for col in range(data.shape[1]):
            window = data[:, col]
            features.extend([
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.median(window),
                np.percentile(window, 25),  # Q1
                np.percentile(window, 75),  # Q3
                np.max(np.abs(np.diff(window))),  # Max rate of change
                np.mean(np.abs(np.diff(window))),  # Mean rate of change
                np.std(np.diff(window))  # Std of rate of change
            ])
        return np.array(features)

    def _calculate_derivatives(self, data):
        """
        Calculate first and second derivatives
        
        Args:
            data (np.array): Time series data
            
        Returns:
            tuple: First and second derivatives
        """
        first_derivative = np.diff(data, axis=0)
        second_derivative = np.diff(first_derivative, axis=0)
        return first_derivative, second_derivative

    def _detect_peaks(self, data, threshold=0.1):
        """
        Detect peaks in the time series
        
        Args:
            data (np.array): Time series data
            threshold (float): Threshold for peak detection
            
        Returns:
            np.array: Peak indicators
        """
        peaks = np.zeros_like(data)
        for col in range(data.shape[1]):
            series = data[:, col]
            for i in range(1, len(series)-1):
                if series[i] > series[i-1] and series[i] > series[i+1] and \
                   series[i] > np.mean(series) + threshold * np.std(series):
                    peaks[i, col] = 1
        return peaks

    def _analyze_trend(self, data):
        """
        Analyze trend in the time series
        
        Args:
            data (np.array): Time series data
            
        Returns:
            np.array: Trend indicators
        """
        trends = np.zeros_like(data)
        for col in range(data.shape[1]):
            series = data[:, col]
            # Calculate moving average
            ma = pd.Series(series).rolling(window=5).mean().fillna(method='bfill')
            # Calculate trend (1 for increasing, -1 for decreasing, 0 for stable)
            diff = np.diff(ma)
            trends[:-1, col] = np.sign(diff)
        return trends

    def preprocess_simulation(self, df):
        """
        Preprocess a single simulation
        
        Args:
            df (pd.DataFrame): Raw simulation data
            
        Returns:
            np.array: Preprocessed features
        """
        # Convert to numpy array
        data = df.values
        
        # Initialize list to store features
        all_features = []
        
        # Process data in windows
        for i in range(0, len(data) - self.window_size + 1):
            window = data[i:i + self.window_size]
            
            # Calculate features
            statistical_features = self._calculate_statistical_features(window)
            first_der, second_der = self._calculate_derivatives(window)
            peaks = self._detect_peaks(window)
            trends = self._analyze_trend(window)
            
            # Combine all features
            window_features = np.concatenate([
                statistical_features,
                first_der[-1].flatten(),  # Use last point of first derivative
                second_der[-1].flatten(),  # Use last point of second derivative
                peaks[-1].flatten(),  # Use last point of peaks
                trends[-1].flatten()  # Use last point of trends
            ])
            
            all_features.append(window_features)
        
        return np.array(all_features)

    def preprocess_dataset(self, data_dir):
        """
        Preprocess the entire dataset
        
        Args:
            data_dir (str): Directory containing the preprocessed data
            
        Returns:
            tuple: (X, y) where X is the features array and y is the labels array
        """
        logging.info("Starting dataset preprocessing")
        
        X = []  # List to store features
        y = []  # List to store labels
        
        # Map accident types to indices
        accident_types = sorted(os.listdir(data_dir))
        accident_to_idx = {acc: idx for idx, acc in enumerate(accident_types)}
        
        # Process each accident type
        for accident_type in tqdm(accident_types, desc="Processing accident types"):
            accident_path = os.path.join(data_dir, accident_type)
            if os.path.isdir(accident_path):
                # Process each simulation
                for simulation in tqdm(os.listdir(accident_path), desc=f"Processing {accident_type}"):
                    if simulation.endswith('.csv'):
                        file_path = os.path.join(accident_path, simulation)
                        try:
                            # Load simulation data
                            df = pd.read_csv(file_path)
                            
                            # Preprocess simulation
                            features = self.preprocess_simulation(df)
                            
                            X.append(features)
                            y.extend([accident_to_idx[accident_type]] * len(features))
                            
                        except Exception as e:
                            logging.error(f"Error processing {file_path}: {str(e)}")
                            continue
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to (samples, time_steps, features)
        X = X.reshape(-1, X.shape[1], X.shape[2])
        
        # Scale features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        logging.info(f"Preprocessing completed. Final shape: {X.shape}")
        return X, y

def main():
    # Initialize preprocessor
    preprocessor = DLPreprocessor(window_size=5)
    
    # Preprocess dataset
    data_dir = 'NPPAD_for_classifiers'
    X, y = preprocessor.preprocess_dataset(data_dir)
    
    # Save preprocessed data
    output_dir = 'preprocessing/classification/DL'
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    logging.info("Preprocessed data saved successfully")

if __name__ == "__main__":
    main() 