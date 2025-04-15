#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Facial expression classification model using CNN architecture.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam


# Mapping of indices to expression labels
EXPRESSION_LABELS = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'sad',
    'surprise',
    'neutral'
]


class ExpressionClassifier:
    """Facial expression classification model."""
    
    def __init__(self, model_path=None, input_shape=(48, 48, 1)):
        """
        Initialize the facial expression classifier.
        
        Args:
            model_path: Path to pre-trained model file
            input_shape: Input shape expected by the model
        """
        self.input_shape = input_shape
        self.model = None
        
        if model_path and os.path.isfile(model_path):
            # Load existing model
            self.model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            # Initialize model architecture if no pre-trained model is provided
            self.model = self._build_model()
            print("Created new model (untrained)")
            
            # Save model directory path
            if model_path:
                self.model_dir = os.path.dirname(model_path)
                os.makedirs(self.model_dir, exist_ok=True)
                print(f"Model will be saved to {model_path}")
    
    def _build_model(self):
        """
        Build the CNN model architecture.
        
        Returns:
            Compiled model
        """
        model = Sequential()
        
        # First Convolutional Block
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', 
                        input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second Convolutional Block
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third Convolutional Block
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Flatten the feature maps
        model.add(Flatten())
        
        # Fully Connected Layers
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Output Layer (7 emotions)
        model.add(Dense(len(EXPRESSION_LABELS), activation='softmax'))
        
        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
             batch_size=64, epochs=50, model_save_path=None):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of epochs to train
            model_save_path: Path to save the trained model
            
        Returns:
            Training history
        """
        if self.model is None:
            self.model = self._build_model()
        
        # Define callbacks
        callbacks = []
        
        # ModelCheckpoint callback to save best model
        if model_save_path:
            from tensorflow.keras.callbacks import ModelCheckpoint
            checkpoint = ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Early stopping callback to prevent overfitting
        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, face_image):
        """
        Predict expression for a face image.
        
        Args:
            face_image: Input face image
            
        Returns:
            Tuple of (expression_label, probabilities_dict)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Load or train a model first.")
        
        # Preprocess the face image
        processed_image = self._preprocess_face(face_image)
        
        # Make prediction
        prediction = self.model.predict(processed_image)[0]
        
        # Get the most likely expression
        expression_idx = np.argmax(prediction)
        expression = EXPRESSION_LABELS[expression_idx]
        
        # Create a dictionary of expression probabilities
        probabilities = {label: float(prob) for label, prob in zip(EXPRESSION_LABELS, prediction)}
        
        return expression, probabilities
    
    def _preprocess_face(self, face_image):
        """
        Preprocess face image for model input.
        
        Args:
            face_image: Input face image
            
        Returns:
            Preprocessed image ready for model input
        """
        # Convert to grayscale if image is colored
        if len(face_image.shape) == 3 and face_image.shape[2] > 1:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_image
        
        # Resize to model input size
        resized_face = cv2.resize(gray_face, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize pixel values to [0, 1]
        normalized_face = resized_face / 255.0
        
        # Reshape to model input shape with batch dimension
        input_face = normalized_face.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        
        return input_face
    
    def save(self, model_path):
        """
        Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Initialize or train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Load or train a model first.")
        
        # Evaluate the model
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Create a dictionary of metrics
        metrics = {metric: value for metric, value in zip(self.model.metrics_names, results)}
        
        return metrics
