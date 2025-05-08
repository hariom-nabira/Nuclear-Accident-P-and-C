# Classification Model Plan

## Overview
The classification model aims to classify time series data of operational parameters into one of the 12 accident types. We will implement both traditional ML models and a deep learning model.

## Models to Implement

### 1. Traditional ML Models

#### Random Forest
- Features: Statistical features from time series windows
- Hyperparameters to tune:
  - n_estimators
  - max_depth
  - min_samples_split
  - min_samples_leaf

#### K-Nearest Neighbors (KNN)
- Features: Statistical features from time series windows
- Hyperparameters to tune:
  - n_neighbors
  - weights
  - metric
  - algorithm

#### XGBoost
- Features: Statistical features from time series windows
- Hyperparameters to tune:
  - learning_rate
  - max_depth
  - n_estimators
  - subsample
  - colsample_bytree

#### Support Vector Machine (SVM)
- Features: Statistical features from time series windows
- Hyperparameters to tune:
  - C
  - kernel
  - gamma
  - class_weight

### 2. Deep Learning Model: Hybrid LSTM+GRU

#### Architecture
```
Input Layer
    ↓
LSTM Layer 1 (128 units)
    ↓
LSTM Layer 2 (64 units)
    ↓
GRU Layer 1 (64 units)
    ↓
GRU Layer 2 (32 units)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense Layer (12 units, Softmax)
```

## Preprocessing Pipeline

### 1. Data Loading and Preparation
- Load CSV files from all accident types
- Load corresponding transient reports
- Align time series data with event timestamps
- Create balanced dataset across accident types

### 2. Feature Engineering
- Time window features (e.g., 30-second windows)
- Statistical features per window:
  - Mean, std, min, max
  - First and second derivatives
  - Rate of change
  - Peak detection
  - Trend analysis
- Domain-specific features:
  - Reactor power level changes
  - Temperature gradients
  - Pressure differentials
  - Flow rate variations

### 3. Data Splitting
- Stratified split by accident type
- 70% training
- 15% validation
- 15% testing

### 4. Data Normalization
- Min-Max scaling for ML models
- Standard scaling for DL model

## Model Training

### ML Models
1. Grid search for hyperparameter tuning
2. Cross-validation (5-fold)
3. Model selection based on:
   - Accuracy
   - F1-score
   - Confusion matrix
   - ROC-AUC

### DL Model
1. Training strategy:
   - Batch size: 32
   - Epochs: 100
   - Early stopping
   - Learning rate scheduling
2. Regularization:
   - Dropout
   - L2 regularization
   - Batch normalization

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- Classification report

## Implementation Steps

1. Data Preparation
   - Create preprocessing scripts
   - Generate feature engineering pipeline
   - Create data loaders

2. Model Development
   - Implement ML models
   - Implement DL model
   - Create training pipelines

3. Model Training
   - Train and tune ML models
   - Train DL model
   - Save best models

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
- scikit-learn
- xgboost
- tensorflow/keras
- pandas
- numpy
- matplotlib
- seaborn

## Logging
- Training metrics
- Model parameters
- Performance metrics
- Error analysis

## Next Steps
1. Set up directory structure
2. Implement preprocessing pipeline
3. Begin with ML model development
4. Implement DL model
5. Conduct experiments and evaluations 