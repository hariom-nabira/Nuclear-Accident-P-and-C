import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import joblib
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'ml_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

class MLModelTrainer:
    def __init__(self, data_dir="NPPAD_for_classifiers_features"):
        """
        Initialize ML model trainer.
        data_dir: directory containing feature-engineered data
        """
        # Get the project root directory (3 levels up from the current file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        self.data_dir = os.path.join(project_root, data_dir)
        
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'knn': KNeighborsClassifier(),
            'xgboost': XGBClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        self.best_models = {}
        self.accident_types = []
        self.label_encoder = LabelEncoder()
        self.metrics = {}
        
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
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, output_dir):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(12, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
    
    def plot_roc_curves(self, y_test, y_score, model_name, output_dir):
        """Plot and save ROC curves for each class."""
        plt.figure(figsize=(12, 8))
        
        # Binarize the output
        n_classes = len(self.label_encoder.classes_)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'black'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {self.label_encoder.classes_[i]} (area = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curves.png'))
        plt.close()
    
    def plot_feature_importance(self, model, model_name, output_dir):
        """Plot and save feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title(f'Feature Importances - {model_name}')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), indices, rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'))
            plt.close()
    
    def save_metrics(self, model_name, metrics, output_dir):
        """Save metrics to CSV file."""
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, f'{model_name}_metrics.csv'), index=False)
    
    def train_models(self, X, y):
        """Train all ML models with hyperparameter tuning."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create output directory for visualizations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "visualizations")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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
            y_score = grid_search.predict_proba(X_test)
            
            # Convert predictions back to original labels for reporting
            y_test_original = self.label_encoder.inverse_transform(y_test)
            y_pred_original = self.label_encoder.inverse_transform(y_pred)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test_original, y_pred_original, output_dict=True)
            
            # Store metrics
            self.metrics[model_name] = {
                'accuracy': accuracy,
                'best_params': grid_search.best_params_,
                'classification_report': report
            }
            
            # Create visualizations
            self.plot_confusion_matrix(y_test_original, y_pred_original, model_name, output_dir)
            self.plot_roc_curves(y_test, y_score, model_name, output_dir)
            self.plot_feature_importance(grid_search.best_estimator_, model_name, output_dir)
            
            # Save metrics
            self.save_metrics(model_name, self.metrics[model_name], output_dir)
            
            # Log results
            logging.info(f"\nBest parameters for {model_name}:")
            logging.info(grid_search.best_params_)
            logging.info(f"\nTest accuracy: {accuracy:.4f}")
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test_original, y_pred_original))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test_original, y_pred_original))
    
    def save_models(self, output_dir=None):
        """Save trained models and label encoder."""
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, "saved_models")
            
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
        
        self.save_models()
        
        logging.info("Training completed!")

def main():
    """Main function to run the training pipeline."""
    trainer = MLModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main() 