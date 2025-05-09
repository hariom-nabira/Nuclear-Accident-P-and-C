import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input, Bidirectional, Concatenate, Attention
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
        Build the enhanced Hybrid LSTM+GRU model architecture
        
        Returns:
            model: Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Bidirectional LSTM branch
        lstm_branch = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = Bidirectional(LSTM(64, return_sequences=True))(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        
        # Bidirectional GRU branch
        gru_branch = Bidirectional(GRU(128, return_sequences=True))(inputs)
        gru_branch = BatchNormalization()(gru_branch)
        gru_branch = Bidirectional(GRU(64, return_sequences=True))(gru_branch)
        gru_branch = BatchNormalization()(gru_branch)
        
        # Attention mechanism
        attention = Attention()([lstm_branch, gru_branch])
        
        # Concatenate branches
        concat = Concatenate()([lstm_branch, gru_branch, attention])
        
        # Final GRU layer
        x = GRU(64, return_sequences=False)(concat)
        x = BatchNormalization()(x)
        
        # Dense layers with residual connections
        dense1 = Dense(128, activation='relu')(x)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.3)(dense2)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(dense2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with custom learning rate
        optimizer = Adam(learning_rate=0.0005)  # Reduced learning rate
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logging.info("Enhanced model architecture built and compiled")
        model.summary(print_fn=logging.info)
        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100):
        """
        Train the Hybrid LSTM+GRU model with enhanced callbacks
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Maximum number of epochs
        """
        # Create enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience
                restore_best_weights=True,
                mode='min'
            ),
            ModelCheckpoint(
                filepath=os.path.join('models', 'classification', 'DL', 'saved_models', 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=7,  # Increased patience
                min_lr=0.00001,
                mode='min'
            ),
            # Add TensorBoard callback for monitoring
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join('models', 'classification', 'DL', 'logs', 'tensorboard'),
                histogram_freq=1
            )
        ]

        # Train model with class weights if needed
        class_weights = self._calculate_class_weights(y_train)
        
        logging.info("Starting model training")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        logging.info("Model training completed")
        return history

    def _calculate_class_weights(self, y_train):
        """
        Calculate class weights to handle class imbalance
        
        Args:
            y_train: Training labels
            
        Returns:
            dict: Class weights
        """
        # Convert one-hot encoded labels to class indices
        y_indices = np.argmax(y_train, axis=1)
        
        # Calculate class weights
        class_counts = np.bincount(y_indices)
        total_samples = len(y_indices)
        class_weights = {
            i: total_samples / (len(class_counts) * count)
            for i, count in enumerate(class_counts)
        }
        
        logging.info(f"Calculated class weights: {class_weights}")
        return class_weights

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