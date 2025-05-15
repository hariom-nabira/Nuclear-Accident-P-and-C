# Nuclear Power Plant Accident Detection and Classification System

## Introduction
Nuclear Power Plants (NPPs) are complex systems that require constant monitoring and early detection of potential accidents to ensure safe operation. This project addresses the critical challenge of real-time accident detection and classification in NPPs using advanced machine learning and deep learning techniques.

The system processes time-series data of operational parameters (collected at 10-second intervals) and has two main components (with a third planned for future scope):
1. **Binary Accident Prediction**: Predicts if an accident will occur in the next 300 seconds
2. **Time Series Forecasting** (Future Scope): Predicts parameter values for the next 300 seconds
3. **Classification Model**: Identifies the type of accident using historical and predicted data

## Project Structure
```
SEM-PROJECT/
├── NPPAD/                          # Original Nuclear Power Plant Accident Dataset
├── NPPAD_for_classifiers/          # Base dataset for classification models
├── NPPAD_for_classifiers_dl/       # Processed dataset for deep learning classifiers
├── NPPAD_for_classifiers_features/ # Feature-engineered dataset for ML classifiers
├── NPPAD_for_prediction/           # Dataset for binary accident prediction
├── models/                         # Trained models
│   ├── classification/            # Classification models
│   │   ├── ML/                   # Machine Learning models
│   │   └── DL/                   # Deep Learning models
│   └── prediction/                # Binary prediction models
├── preprocessing/                  # Data preprocessing scripts
│   ├── classification/            # Classification-specific preprocessing
│   ├── prediction/                # Prediction-specific preprocessing
│   └── standardize_csv.py         # Common data standardization
├── logs/                          # Execution logs and metrics
├── documentation/                  # Project documentation and plans
└── requirements.txt               # Project dependencies
```

## Features

### 1. Binary Accident Prediction
- Implements Temporal Convolutional Network (TCN) with attention mechanism
- Predicts accident occurrence within 300 seconds (30 time steps)
- Focuses on critical events:
  - Reactor Scram
  - Core Meltdown
- Real-time prediction capability
- High accuracy in early warning

### 2. Classification System
- Multiple model implementations:
  - Machine Learning Models:
    - Random Forest
    - KNN
    - XGBoost
    - SVM
  - Deep Learning Model:
    - Hybrid LSTM+GRU architecture
- Classifies 12 different types of accidents
- Uses time-series data of operational parameters
- Separate feature engineering for ML and DL approaches

### 3. Data Processing Pipeline
- Comprehensive preprocessing workflow:
  - Data standardization
  - Feature engineering
  - Time-series window creation
  - Data validation and cleaning
- Separate preprocessing for:
  - Classification models
  - Binary prediction models
- Automated logging of preprocessing steps

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hariom-nabira/Nuclear-Accident-P-and-C.git
cd Nuclear-Accident-P-and-C
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
1. Place the NPPAD dataset in the `NPPAD/` directory
2. Run preprocessing scripts:
```bash
# Standardize the dataset
python preprocessing/standardize_csv.py

# Prepare data for classification
python preprocessing/classification/prepare_classification_data.py

# Prepare data for binary prediction
python preprocessing/prediction/prepare_prediction_data.py
```

### Model Training
1. For classification models:
```bash
# Train ML classifiers
python models/classification/ML/ml_models.py

# Train DL classifier
python models/classification/DL/train_hybrid_model.py
```

2. For binary prediction:
```bash
python models/prediction/tcn_attention.py
```

## Dependencies
- Python 3.8+
- Key packages:
  - Data processing: pandas, numpy
  - Machine Learning: scikit-learn, xgboost
  - Deep Learning: tensorflow, keras
  - Visualization: matplotlib, seaborn
  - Utilities: tqdm, python-dotenv, joblib

See `requirements.txt` for complete list of dependencies.

## Documentation
- `documentation/plan.md`: Overall project plan
- `documentation/classification_plan.md`: Classification system details
- `documentation/binary_prediction_plan.md`: Binary prediction system details
- `documentation/forecasting_plan.md`: Future forecasting system plan
- Working documentation for each component in `documentation/` subdirectories

## Logging
- All scripts generate logs in the `logs/` directory
- Log files are organized by component and timestamp
- Includes:
  - Training metrics
  - Preprocessing steps
  - Model evaluation results
  - Error logs

## Future Scope
1. Implementation of Time Series Forecasting system
2. Real-time prediction pipeline
3. System integration and monitoring
4. Performance optimization
5. Additional model architectures
6. Integration with plant control systems
7. Real-time visualization dashboard
