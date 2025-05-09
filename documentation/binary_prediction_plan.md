# Binary Accident Prediction Model Plan

## Overview
The binary prediction model aims to predict whether an accident (Reactor Scram or Core Meltdown) will occur within the next 300 seconds based on the time series data of operational parameters.

## Model Architecture: TCN with Attention

### Temporal Convolutional Network (TCN)
```
Input Layer (Time series data)
    ↓
Causal Convolutional Block 1
    - Dilation rate: 1
    - Filters: 64
    - Kernel size: 3
    ↓
Causal Convolutional Block 2
    - Dilation rate: 2
    - Filters: 64
    - Kernel size: 3
    ↓
Causal Convolutional Block 3
    - Dilation rate: 4
    - Filters: 64
    - Kernel size: 3
    ↓
Causal Convolutional Block 4
    - Dilation rate: 8
    - Filters: 64
    - Kernel size: 3
    ↓
Attention Layer
    - Multi-head attention (4 heads)
    - Attention dimension: 64
    ↓
Global Average Pooling
    ↓
Dense Layer (32 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense Layer (1 unit, Sigmoid)
```

### Key Components
1. Causal Convolutional Blocks
   - Residual connections
   - Batch normalization
   - ReLU activation
   - Dropout (0.2)

2. Attention Mechanism
   - Self-attention
   - Position-wise feed-forward network
   - Layer normalization

## Preprocessing Pipeline

### 1. Data Loading and Preparation
- Load CSV files and transient reports
- Identify accident events (Reactor Scram or Core Meltdown)
- Create binary labels (1 for accident within 300s, 0 otherwise)
- Handle class imbalance if necessary

### 2. Feature Engineering
- Time window features (300-second lookback)
- Rate of change features
- Statistical features:
  - Rolling mean
  - Rolling std
  - Rolling min/max
- Domain-specific features:
  - Power level changes
  - Temperature gradients
  - Pressure differentials
  - Flow rate variations

### 3. Data Splitting
- Stratified split by accident type
- 70% training
- 15% validation
- 15% testing

### 4. Data Normalization
- Standard scaling for all features
- Sequence padding if necessary

## Model Training

### Training Strategy
1. Batch size: 32
2. Epochs: 100
3. Early stopping with patience=10
4. Learning rate scheduling:
   - Initial rate: 0.001
   - Reduce on plateau
   - Minimum rate: 1e-6

### Loss Function
- Binary Cross Entropy
- Class weights to handle imbalance

### Regularization
- Dropout
- L2 regularization
- Batch normalization

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall curve
- Confusion matrix

## Implementation Steps

1. Data Preparation
   - Create preprocessing scripts
   - Generate feature engineering pipeline
   - Create data loaders

2. Model Development
   - Implement TCN architecture
   - Implement attention mechanism
   - Create training pipeline

3. Model Training
   - Train model
   - Hyperparameter tuning
   - Save best model

4. Evaluation
   - Evaluate on test set
   - Generate performance metrics
   - Create visualizations

5. Documentation
   - Model architecture
   - Training process
   - Performance results
   - Usage instructions

## Dependencies
- tensorflow/keras
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Logging
- Training metrics
- Model parameters
- Performance metrics
- Error analysis
- Prediction confidence scores

## Next Steps
1. Set up directory structure
2. Implement preprocessing pipeline
3. Develop TCN model
4. Implement attention mechanism
5. Conduct experiments and evaluations 