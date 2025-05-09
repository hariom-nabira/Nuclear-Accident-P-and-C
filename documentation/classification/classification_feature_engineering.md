# Classification Feature Engineering Documentation

## Overview
This document describes the feature engineering process for the classification task. The goal is to extract meaningful features from the time series data that will help in classifying different types of accidents.

## Feature Engineering Steps

### 1. Statistical Features
For each time window (default: 50 seconds):
- Mean
- Standard deviation
- Minimum
- Maximum
- Rate of change (first derivative)
- Acceleration (second derivative)

### 2. Feature Normalization
- Min-Max scaling for all features
- Scale range: [0, 1]
- Separate scaler for each simulation

## Implementation Details

### Input/Output
- Input: `NPPAD_for_classifiers/`
  - Preprocessed CSV files
  - Each file contains time series data up to accident
- Output: `NPPAD_for_classifiers_features/`
  - Processed CSV files with engineered features
  - Same directory structure as input
  - Normalized features

### Processing Steps
1. Read each simulation file
2. Calculate statistical features:
   - Rolling window statistics (mean, std, min, max)
   - First derivative (rate of change)
   - Second derivative (acceleration)
3. Normalize features
4. Save processed data

### Error Handling
- Missing values are filled using backward fill
- Invalid calculations are logged
- Processing errors are logged and skipped

## Usage
```bash
python preprocessing/model_specific/classification/feature_engineering.py
```

## Feature Set
For each original parameter (except time), we generate 6 statistical features:
- Mean (rolling window)
- Standard deviation (rolling window)
- Minimum (rolling window)
- Maximum (rolling window)
- Rate of change (first derivative)
- Acceleration (second derivative)

## Implementation Notes
- Window size is set to 50 seconds by default
- All features are calculated using pandas rolling window operations
- Missing values at the start of the time series are filled using backward fill
- Each simulation is normalized independently

## Next Steps
1. Verify feature engineering results:
   - Check feature distributions
   - Validate normalization
   - Ensure no information loss
2. Begin model development:
   - Implement ML models
   - Implement DL model
3. Create data loaders for model training 