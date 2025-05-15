# Hybrid LSTM+GRU Model Documentation

## Overview
This document describes the implementation of the Hybrid LSTM+GRU model for nuclear accident classification. The model combines bidirectional LSTM and GRU layers with an attention mechanism to capture both long-term and short-term dependencies in the time series data.

## Model Architecture

### Layer Structure
1. Input Layer
   - Accepts time series data with shape (time_steps, features)

2. Parallel LSTM Branch
   - First Bidirectional LSTM layer: 128 units with return sequences
   - Batch Normalization
   - Second Bidirectional LSTM layer: 64 units with return sequences
   - Batch Normalization

3. Parallel GRU Branch
   - First Bidirectional GRU layer: 128 units with return sequences
   - Batch Normalization
   - Second Bidirectional GRU layer: 64 units with return sequences
   - Batch Normalization

4. Attention Mechanism
   - Attention layer combining LSTM and GRU outputs

5. Feature Fusion
   - Concatenation of LSTM, GRU, and attention outputs

6. Final Processing
   - GRU layer: 64 units without return sequences
   - Batch Normalization
   - Dense layer: 128 units with ReLU activation
   - Batch Normalization
   - Dropout (0.4)
   - Dense layer: 64 units with ReLU activation
   - Batch Normalization
   - Dropout (0.3)
   - Output layer: num_classes units with softmax activation

### Regularization Techniques
- Batch Normalization after each recurrent layer
- Dropout (0.4 and 0.3) in dense layers
- Attention mechanism for feature importance

## Training Process

### Data Preparation
1. Data Loading
   - Loads preprocessed data
   - Converts labels to one-hot encoding
   - Splits data into train, validation, and test sets

2. Data Normalization
   - Standard scaling of features

### Training Configuration
- Batch size: 32
- Maximum epochs: 100
- Optimizer: Adam with learning rate 0.0005
- Loss function: Categorical Crossentropy
- Metrics: Accuracy

### Callbacks
1. Early Stopping
   - Monitors validation loss
   - Patience: 15 epochs
   - Restores best weights

2. Model Checkpoint
   - Saves best model based on validation accuracy
   - Saves to 'best_model.h5'

3. ReduceLROnPlateau
   - Reduces learning rate when validation loss plateaus
   - Factor: 0.2
   - Patience: 7 epochs
   - Minimum learning rate: 0.00001

4. TensorBoard
   - Logs training metrics and model architecture
   - Enables visualization of training progress

### Class Imbalance Handling
- Implements dynamic class weighting
- Weights are calculated based on class distribution
- Helps balance the training process for imbalanced datasets

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
- TensorBoard logs for visualization

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
5. Optimize hyperparameters for better performance 