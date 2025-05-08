# NPP Accident Prediction and Classification System

## Overview
This system aims to predict and classify nuclear power plant accidents using time-series data of operational parameters. The system consists of three main components:

1. Classification Model
2. Binary Accident Prediction Model
3. Time Series Forecasting Model

## System Architecture

### Data Flow
1. Input: Continuous stream of operational parameters at 10-second intervals
2. Binary Prediction: Predicts if an accident will occur in next 300 seconds
3. If accident predicted:
   - Time Series Forecasting: Predicts parameter values for next 300 seconds
   - Classification: Classifies the accident type using both historical and predicted data

### Dataset Structure
- Base dataset: NPPAD
- 12 accident types (subfolders)
- ~100 simulations per accident type
- Each simulation contains:
  - CSV file: Time-series data of operational parameters
  - Transient report: Time stamps of major events

## Component-wise Plans

### 1. Classification Model
- Models to implement:
  - ML Models: Random Forest, KNN, XGBoost, SVM
  - DL Model: Hybrid LSTM+GRU
- See `classification_plan.md` for detailed implementation plan

### 2. Binary Accident Prediction
- Model: TCN with attention mechanism
- Predicts accident occurrence within 300 seconds
- Accident defined as Reactor Scram or Core Meltdown
- See `binary_prediction_plan.md` for detailed implementation plan

### 3. Time Series Forecasting
- Forecasts operational parameters for next 300 seconds (30 steps)
- See `forecasting_plan.md` for detailed implementation plan

## Common Infrastructure

### Directory Structure
```
SEM-PROJECT/
├── NPPAD/                      # Original dataset
├── NPPAD_for_classifiers/      # Dataset for classification models
├── NPPAD_for_prediction/       # Dataset for prediction models
├── models/
│   ├── classification/
│   ├── binary_prediction/
│   └── forecasting/
├── preprocessing/
│   ├── common/
│   │   ├── classifier_preprocessing.py  # Common preprocessing for classifiers
│   │   └── prediction_preprocessing.py  # Common preprocessing for prediction
│   └── model_specific/
├── logs/
├── documentation/
│   ├── working/
│   └── model_specific/
└── requirements.txt
```

### Common Preprocessing Steps

#### 1. Classifier-specific Preprocessing
- Create NPPAD_for_classifiers by cloning NPPAD
- For each simulation:
  - Read transient report to find accident timestamp
  - Drop rows with time > accident_timestamp from corresponding CSV
  - Save modified CSV in NPPAD_for_classifiers
- This preprocessing is common to all classification models

#### 2. General Data Loading and Validation
- Load CSV files and transient reports
- Validate data consistency
- Handle missing values

2. Feature Engineering
   - Time-based features
   - Statistical features
   - Domain-specific features

3. Data Splitting
   - Train/Validation/Test split
   - Time-based splitting to prevent data leakage

4. Data Normalization/Standardization
   - Min-Max scaling
   - Standard scaling
   - Robust scaling

### Logging and Documentation
- Each script will generate logs in the logs folder
- Each component will have its working.md in documentation/working
- Model-specific documentation in documentation/model_specific

### Requirements Management
- requirements.txt will be updated with all necessary dependencies
- Version control for reproducibility

## Implementation Phases

### Phase 1: Data Preparation
1. Set up directory structure
2. Implement common preprocessing pipeline
   - Create NPPAD_for_classifiers with accident-based truncation
   - Create NPPAD_for_prediction for prediction models
3. Create model-specific preprocessing if needed

### Phase 2: Model Development
1. Implement and train Classification models
2. Implement and train Binary Prediction model
3. Implement and train Time Series Forecasting model

### Phase 3: Integration
1. Combine models into unified system
2. Implement real-time prediction pipeline
3. Add monitoring and logging

### Phase 4: Testing and Validation
1. Unit testing for each component
2. Integration testing
3. Performance evaluation
4. System validation

## Next Steps
1. Review and approve individual model plans
2. Set up development environment
3. Begin with data preparation phase 