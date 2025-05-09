import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging
from datetime import datetime

# Set up logging
log_dir = os.path.join('logs', 'classification', 'DL')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'hybrid_lstm_gru_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class HybridLSTMGRU:
    def __init__(self, input_shape, num_classes=12):
        """
        Initialize the Hybrid LSTM+GRU model
        
        Args:
            input_shape (tuple): Shape of input data (time_steps, features)
            num_classes (int): Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        logging.info(f"Model initialized with input shape: {input_shape} and {num_classes} classes")

    def _build_model(self):
        """
        Build the Hybrid LSTM+GRU model architecture
        
        Returns:
            model: Compiled Keras model
        """
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, input_shape=self.input_shape,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            
            # First GRU layer
            GRU(64, return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            
            # Second GRU layer
            GRU(32, return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            
            # Dense layers
            Dense(64, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logging.info("Model architecture built and compiled")
        model.summary(print_fn=logging.info)
        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100):
        """
        Train the Hybrid LSTM+GRU model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Maximum number of epochs
        """
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join('models', 'classification', 'DL', 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001
            )
        ]

        # Train model
        logging.info("Starting model training")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logging.info("Model training completed")
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test data
            y_test: Test labels
        """
        logging.info("Evaluating model on test data")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test accuracy: {test_accuracy:.4f}")
        logging.info(f"Test loss: {test_loss:.4f}")
        return test_loss, test_accuracy

    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input data
        """
        return self.model.predict(X)

    def save_model(self, filepath):
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        logging.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a saved model from disk
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        logging.info(f"Model loaded from {filepath}") 