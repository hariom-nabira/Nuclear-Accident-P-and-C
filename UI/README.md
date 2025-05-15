# Nuclear Accident Prediction and Classification System

This Streamlit application provides a user interface for predicting nuclear reactor scrams and classifying potential accidents using machine learning models.

## Features

- Upload and analyze nuclear reactor data in CSV format
- Real-time prediction of reactor scrams using TCN-Attention model
- Accident classification using Hybrid LSTM-GRU model when scram is predicted
- Interactive data visualization and results display

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Input Data Format

The application expects CSV files with the following structure:
- Time-series data with reactor parameters
- Similar format to the reference dataset in NPPAD/FLB/1.csv

## Models

The application uses two main models:
1. TCN-Attention model for scram prediction
2. Hybrid LSTM-GRU model for accident classification

## Usage

1. Launch the application using `streamlit run app.py`
2. Upload your CSV file using the file uploader
3. Review the data preview
4. Click "Start Analysis" to begin processing
5. View results and predictions in real-time 