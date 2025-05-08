# Classification Preprocessing Documentation

## Overview
This document describes the preprocessing steps for the classification task. The main goal is to prepare the dataset by truncating time series data at the point of accident occurrence.

## Preprocessing Steps

### 1. Common Preprocessing (classifier_preprocessing.py)
- Creates a new dataset `NPPAD_for_classifiers` by cloning `NPPAD`
- For each simulation:
  - Reads the transient report to find accident timestamp (Reactor Scram or Core Meltdown)
  - Truncates the CSV data at the accident timestamp
  - Saves the preprocessed data in the new dataset
  - Logs the accident type and timestamp for verification

### 2. File Structure
- Input: `NPPAD/`
  - Contains original dataset with 12 accident types
  - Each accident type has varying number of simulations
  - Each simulation has a CSV file and a transient report
- Output: `NPPAD_for_classifiers/`
  - Mirrors the structure of `NPPAD/`
  - Contains preprocessed CSV files
  - Original transient reports are copied as is

### 3. Logging
- Logs are stored in `logs/classifier_preprocessing_YYYYMMDD_HHMMSS.log`
- Logs include:
  - Number of simulations found for each accident type
  - Processing status for each accident type
  - Success/failure of each simulation preprocessing
  - Accident type and timestamp for each processed simulation
  - Any errors or warnings encountered

### 4. Error Handling
- Missing files are logged and skipped
- Invalid transient reports are logged and skipped
- CSV parsing errors are logged and skipped
- Simulations without accidents are kept as is (not skipped)

### 5. File Processing
- Dynamically discovers all simulation files in each accident directory
- Matches CSV files with corresponding transient reports
- Sorts simulations by severity number for consistent processing
- Processes all valid simulation pairs (CSV + transient report):
  - Truncates data at accident timestamp if accident found
  - Keeps original data if no accident found

## Usage
```bash
python preprocessing/common/classifier_preprocessing.py
```

## Output Format
- Each preprocessed CSV file contains:
  - All original columns
  - For simulations with accidents:
    - Data truncated at the accident timestamp
    - Time column in seconds
    - No data after accident occurrence
  - For simulations without accidents:
    - Complete original data
    - No modifications made

## Next Steps
1. Verify the preprocessed dataset:
   - Check number of simulations per accident type
   - Verify accident timestamps
   - Validate data truncation
2. Begin feature engineering for classification models
3. Implement data loading and preparation for specific models 