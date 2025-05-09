# Hybrid LSTM+GRU Model Documentation

## Overview
This document describes the implementation of the Hybrid LSTM+GRU model for nuclear accident classification. The model combines LSTM and GRU layers to capture both long-term and short-term dependencies in the time series data.

## Model Architecture

### Layer Structure
1. Input Layer
   - Accepts time series data with shape (time_steps, features)

2. LSTM Layers
   - First LSTM layer: 128 units with return sequences
   - Second LSTM layer: 64 units with return sequences
   - Both layers include BatchNormalization and L2 regularization

3. GRU Layers
   - First GRU layer: 64 units with return sequences
   - Second GRU layer: 32 units without return sequences
   - Both layers include BatchNormalization and L2 regularization

4. Dense Layers
   - Dense layer: 64 units with ReLU activation
   - Dropout layer: 0.3 dropout rate
   - Output layer: 12 units with softmax activation (one for each accident type)

### Regularization Techniques
- L2 regularization (0.01) on all layers
- Batch Normalization after each recurrent layer
- Dropout (0.3) before the final layer

## Training Process

### Data Preparation
1. Data Loading
   - Loads preprocessed data from NPPAD_for_classifiers
   - Converts labels to one-hot encoding
   - Splits data into train (70%), validation (15%), and test (15%) sets

2. Data Normalization
   - Standard scaling of features

### Training Configuration
- Batch size: 32
- Maximum epochs: 100
- Optimizer: Adam with learning rate 0.001
- Loss function: Categorical Crossentropy
- Metrics: Accuracy

### Callbacks
1. Early Stopping
   - Monitors validation loss
   - Patience: 10 epochs
   - Restores best weights

2. Model Checkpoint
   - Saves best model based on validation accuracy
   - Saves to 'best_model.h5'

3. ReduceLROnPlateau
   - Reduces learning rate when validation loss plateaus
   - Factor: 0.2
   - Patience: 5 epochs
   - Minimum learning rate: 0.00001

## Model Evaluation
- Evaluates on test set
- Metrics:
  - Accuracy
  - Loss
  - Confusion Matrix
  - Classification Report

## Usage

### Training
```python
from hybrid_lstm_gru import HybridLSTMGRU

# Initialize model
model = HybridLSTMGRU(input_shape=(time_steps, features), num_classes=12)

# Train model
history = model.train(X_train, y_train, X_val, y_val)
```

### Prediction
```python
# Make predictions
predictions = model.predict(X_test)
```

### Model Saving/Loading
```python
# Save model
model.save_model('path/to/save/model.h5')

# Load model
model.load_model('path/to/load/model.h5')
```

## Logging
- Training logs are saved in 'logs/classification/DL/'
- Each training session creates a new log file with timestamp
- Logs include:
  - Model architecture
  - Training progress
  - Evaluation metrics
  - Error messages (if any)

## Dependencies
- TensorFlow >= 2.8.0
- NumPy
- Pandas
- scikit-learn

## Next Steps
1. Implement data loading logic specific to NPPAD_for_classifiers
2. Add visualization of training metrics
3. Implement model evaluation metrics
4. Add model interpretation tools 