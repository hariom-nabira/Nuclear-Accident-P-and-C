import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/ml_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class MLModelTrainer:
    def __init__(self, data_dir="NPPAD_for_classifiers_features"):
        """
        Initialize ML model trainer.
        data_dir: directory containing feature-engineered data
        """
        self.data_dir = data_dir
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'knn': KNeighborsClassifier(),
            'xgboost': XGBClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        self.best_models = {}
        self.accident_types = []
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and prepare data for training."""
        X = []
        y = []
        
        # Load data from each accident type
        for accident_type in os.listdir(self.data_dir):
            accident_dir = os.path.join(self.data_dir, accident_type)
            if not os.path.isdir(accident_dir):
                continue
                
            logging.info(f"Loading data for accident type: {accident_type}")
            self.accident_types.append(accident_type)
            
            # Load each simulation
            for csv_file in tqdm(os.listdir(accident_dir), desc=f"Loading {accident_type}"):
                if not csv_file.endswith('.csv'):
                    continue
                    
                # Read features
                df = pd.read_csv(os.path.join(accident_dir, csv_file))
                
                # Use the last row of each simulation as features
                X.append(df.iloc[-1].values)
                y.append(accident_type)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        logging.info(f"Class mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return np.array(X), y_encoded
    
    def train_models(self, X, y):
        """Train all ML models with hyperparameter tuning."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define hyperparameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        }
        
        # Train each model
        for model_name, model in self.models.items():
            logging.info(f"Training {model_name}...")
            
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Save best model
            self.best_models[model_name] = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test)
            
            # Convert predictions back to original labels for reporting
            y_test_original = self.label_encoder.inverse_transform(y_test)
            y_pred_original = self.label_encoder.inverse_transform(y_pred)
            
            # Log results
            logging.info(f"\nBest parameters for {model_name}:")
            logging.info(grid_search.best_params_)
            logging.info(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test_original, y_pred_original))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test_original, y_pred_original))
    
    def save_models(self, output_dir="models/classification/saved_models"):
        """Save trained models and label encoder."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for model_name, model in self.best_models.items():
            model_path = os.path.join(output_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            logging.info(f"Saved {model_name} to {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, "label_encoder.joblib")
        joblib.dump(self.label_encoder, encoder_path)
        logging.info(f"Saved label encoder to {encoder_path}")
    
    def run(self):
        """Run the complete training pipeline."""
        # Load data
        X, y = self.load_data()
        
        # Train models
        self.train_models(X, y)
        
        # Save models
        self.save_models()
        
        logging.info("Training completed!")

def main():
    """Main function to run the training pipeline."""
    trainer = MLModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main() 