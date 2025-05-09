import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from hybrid_lstm_gru import HybridLSTMGRU
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Create necessary directories
base_dir = os.path.dirname(os.path.abspath(__file__))  # models/classification/DL
log_dir = os.path.join(base_dir, 'logs')
saved_models_dir = os.path.join(base_dir, 'saved_models')
visualizations_dir = os.path.join(base_dir, 'visualizations')

for directory in [log_dir, saved_models_dir, visualizations_dir]:
    os.makedirs(directory, exist_ok=True)

# Set up logging
log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def plot_training_history(history, save_dir):
    """
    Plot and save training history
    
    Args:
        history: Training history object
        save_dir: Directory to save plots
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_dir: Directory to save plot
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Convert one-hot encoded predictions to class labels
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.close()

def load_preprocessed_data(data_dir):
    """
    Load preprocessed data from NPPAD_for_classifiers_dl
    
    Args:
        data_dir: Directory containing the preprocessed data
        
    Returns:
        X: Features array
        y: Labels array
    """
    logging.info("Loading preprocessed data")
    
    X_data = []
    y_data = []
    
    # Process each accident type folder
    for accident_type in os.listdir(data_dir):
        accident_path = os.path.join(data_dir, accident_type)
        if os.path.isdir(accident_path):
            # Process each simulation
            for simulation in os.listdir(accident_path):
                if simulation.endswith('.csv'):
                    file_path = os.path.join(accident_path, simulation)
                    
                    # Load preprocessed data
                    df = pd.read_csv(file_path)
                    
                    # Add to data lists
                    X_data.append(df.values)
                    y_data.append(accident_type)
    
    # Convert to numpy arrays
    X = np.array(X_data)
    
    # Convert string labels to integers
    unique_labels = np.unique(y_data)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y_int = np.array([label_to_int[label] for label in y_data])
    
    # Convert to one-hot encoding
    y = tf.keras.utils.to_categorical(y_int, num_classes=len(unique_labels))
    
    logging.info(f"Data loaded: {X.shape[0]} samples with shape {X.shape[1:]} features")
    logging.info(f"Number of classes: {len(unique_labels)}")
    return X, y, unique_labels

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load preprocessed data
    data_dir = 'NPPAD_for_classifiers_dl'
    X, y, unique_labels = load_preprocessed_data(data_dir)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logging.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Initialize and train model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    model = HybridLSTMGRU(input_shape=input_shape, num_classes=y.shape[1])
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=100
    )
    
    # Plot training history
    plot_training_history(history, visualizations_dir)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Get predictions for confusion matrix
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, visualizations_dir)
    
    # Save model
    model_save_path = os.path.join(saved_models_dir, f'hybrid_lstm_gru_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
    model.save_model(model_save_path)
    
    # Save model metadata
    metadata = {
        'input_shape': input_shape,
        'num_classes': y.shape[1],
        'class_labels': unique_labels.tolist(),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss)
    }
    
    import json
    with open(os.path.join(saved_models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logging.info("Training completed successfully")

if __name__ == "__main__":
    main()