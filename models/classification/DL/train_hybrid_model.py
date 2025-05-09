import os
import numpy as np
import tensorflow as tf
from hybrid_lstm_gru import HybridLSTMGRU
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

# Set up logging
log_dir = os.path.join('logs', 'classification', 'DL')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def load_preprocessed_data():
    """
    Load the preprocessed data
    
    Returns:
        tuple: (X, y) where X is the features array and y is the labels array
    """
    logging.info("Loading preprocessed data")
    
    # Load preprocessed data
    data_dir = 'preprocessing/classification/DL'
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    
    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=12)
    
    logging.info(f"Data loaded: {X.shape[0]} samples with {X.shape[2]} features")
    return X, y

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load preprocessed data
    X, y = load_preprocessed_data()
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logging.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Initialize and train model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    model = HybridLSTMGRU(input_shape=input_shape, num_classes=12)
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=100
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Save model
    model_save_path = os.path.join('models', 'classification', 'DL', 'hybrid_lstm_gru_model.h5')
    model.save_model(model_save_path)
    
    logging.info("Training completed successfully")

if __name__ == "__main__":
    main()