# ML Models for Nuclear Power Plant Accident Classification

This document describes the implementation of traditional machine learning models for classifying nuclear power plant accidents.

## Overview

The `ml_models.py` script implements and trains four different machine learning models:
1. Random Forest Classifier
2. K-Nearest Neighbors (KNN)
3. XGBoost
4. Support Vector Machine (SVM)

## Implementation Details

### Data Loading
- The script loads feature-engineered data from the `NPPAD_for_classifiers_features` directory
- For each accident type, it processes all simulation files
- Uses the last row of each simulation as features for classification
- Maintains a list of unique accident types for label encoding

### Model Training
- Implements a `MLModelTrainer` class that handles the complete training pipeline
- Uses 80-20 train-test split with stratification
- Performs hyperparameter tuning using GridSearchCV with 5-fold cross-validation
- Saves the best model for each algorithm

### Hyperparameter Grids

#### Random Forest
- n_estimators: [100, 200, 300]
- max_depth: [10, 20, 30, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

#### KNN
- n_neighbors: [3, 5, 7, 9]
- weights: ['uniform', 'distance']
- metric: ['euclidean', 'manhattan']

#### XGBoost
- n_estimators: [100, 200, 300]
- max_depth: [3, 5, 7]
- learning_rate: [0.01, 0.1, 0.3]
- subsample: [0.8, 0.9, 1.0]

#### SVM
- C: [0.1, 1, 10]
- kernel: ['rbf', 'poly']
- gamma: ['scale', 'auto', 0.1, 1]

### Model Evaluation
- Evaluates each model using:
  - Accuracy score
  - Classification report (precision, recall, F1-score)
  - Confusion matrix
- Logs all evaluation metrics and best parameters

### Model Storage
- Saves trained models in the `models/classification/saved_models` directory
- Uses joblib for model serialization
- Each model is saved with its name as the filename

## Usage

To train the models:
```python
from models.classification.ml_models import MLModelTrainer

trainer = MLModelTrainer()
trainer.run()
```

## Logging

- All training and evaluation logs are saved in the `logs` directory
- Log files are named with timestamp: `ml_models_YYYYMMDD_HHMMSS.log`
- Logs include:
  - Training progress
  - Best parameters for each model
  - Evaluation metrics
  - Model saving confirmation

## Dependencies

- scikit-learn
- pandas
- numpy
- xgboost
- joblib
- tqdm 