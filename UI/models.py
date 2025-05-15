import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Activation, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate):
    """TCN residual block with dilated causal convolutions"""
    prev_x = x
    
    # Layer normalization
    x = LayerNormalization()(x)
    
    # Dilated causal convolution
    x = Conv1D(filters=nb_filters,
               kernel_size=kernel_size,
               padding='causal',
               dilation_rate=dilation_rate,
               activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Second dilated causal convolution
    x = Conv1D(filters=nb_filters,
               kernel_size=kernel_size,
               padding='causal',
               dilation_rate=dilation_rate,
               activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # If dimensions don't match, transform the input
    if prev_x.shape[-1] != nb_filters:
        prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    
    # Residual connection
    res = prev_x + x
    return res

def attention_block(x, num_heads, key_dim):
    """Multi-head self-attention block"""
    # Self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(x, x)
    
    # Skip connection
    return x + attention_output

def build_tcn_attention_model(input_shape):
    """Build TCN model with attention mechanism"""
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Model configuration
    tcn_filters = [64, 128, 128]
    tcn_kernel_size = 3
    tcn_dilations = [1, 2, 4, 8]
    attention_heads = 4
    dropout_rate = 0.3
    
    # TCN blocks with increasing dilation rates
    for i, (nb_filters, dilation_rate) in enumerate(
            zip(tcn_filters, tcn_dilations)):
        x = residual_block(
            x, 
            dilation_rate=dilation_rate,
            nb_filters=nb_filters,
            kernel_size=tcn_kernel_size,
            dropout_rate=dropout_rate
        )
    
    # Attention mechanism
    x = attention_block(x, attention_heads, key_dim=tcn_filters[-1]//attention_heads)
    
    # Global pooling to reduce sequence dimension
    x = GlobalAveragePooling1D()(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    return model

class ModelHandler:
    def __init__(self):
        self.tcn_model = None
        self.hybrid_model = None
        self.scaler = None
        self.sequence_length = 18  # 3 minutes of history (10sec intervals)
        self.feature_cols = None
        
    def load_models(self):
        """Load both TCN-Attention and Hybrid models"""
        try:
            # Custom objects to handle TensorFlow operators
            custom_objects = {
                'TFOpLambda': tf.keras.layers.Layer,
                'tf.__operators__.add': tf.keras.layers.Add(),
                'tf.math.reduce_mean': tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x)),
                'tf.math.multiply': tf.keras.layers.Lambda(lambda x: tf.math.multiply(*x) if isinstance(x, (list, tuple)) else x)
            }
            
            # Load TCN-Attention model from UI/saved_models
            model_dir = Path("saved_models")  # Relative to UI directory
            if (model_dir / "model_fold_1.h5").exists():
                print("Loading model_fold_1.h5...")
                self.tcn_model = tf.keras.models.load_model(
                    str(model_dir / "model_fold_1.h5"),
                    custom_objects=custom_objects
                )
            elif (model_dir / "best_model.h5").exists():
                print("Loading best_model.h5...")
                self.tcn_model = tf.keras.models.load_model(
                    str(model_dir / "best_model.h5"),
                    custom_objects=custom_objects
                )
            else:
                print("No saved model found, building new model...")
                # Build a new model if no saved model is found
                self.tcn_model = build_tcn_attention_model(input_shape=(self.sequence_length, 96))
            
            # For now, we'll use the same model for both tasks
            self.hybrid_model = self.tcn_model
            
            # Initialize scaler
            self.scaler = MinMaxScaler()
            
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def preprocess_data(self, df):
        """Preprocess input dataframe for prediction"""
        if self.feature_cols is None:
            # Skip non-feature columns
            non_feature_cols = ['TIME', 'label', 'accident_timestamp', 'accident_type']
            self.feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        # Extract features
        features = df[self.feature_cols].values
        
        # Scale features
        if self.scaler is not None:
            features = self.scaler.fit_transform(features)  # Using fit_transform since we don't have a saved scaler
        
        return features
    
    def create_sequences(self, features):
        """Create sequences for model input"""
        sequences = []
        for i in range(len(features) - self.sequence_length + 1):
            sequences.append(features[i:i + self.sequence_length])
        return np.array(sequences)
    
    def predict_scram(self, sequence):
        """Predict reactor scram using TCN-Attention model"""
        if self.tcn_model is None:
            raise ValueError("TCN model not loaded")
        
        # Ensure sequence has correct shape
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
        
        try:
            # Make prediction with error handling
            prediction = self.tcn_model.predict(sequence, verbose=0)  # Disable prediction verbosity
            return float(prediction[0][0])  # Return probability of scram
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return 0.0
    
    def classify_accident(self, sequence):
        """Classify accident type using Hybrid model"""
        if self.hybrid_model is None:
            return np.array([1.0, 0.0])  # Return dummy prediction if model not available
        
        # Ensure sequence has correct shape
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
        
        try:
            # Make prediction with error handling
            prediction = self.hybrid_model.predict(sequence, verbose=0)  # Disable prediction verbosity
            return prediction[0]  # Return probabilities for each accident type
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return np.array([1.0, 0.0]) 