#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved facial expression classification model.
This version supports pre-trained models and has better accuracy.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Expression labels
EXPRESSION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class ImprovedExpressionClassifier:
    """Improved facial expression classification model."""
    
    def __init__(self, model_path=None, input_shape=(48, 48, 1), confidence_threshold=0.4):
        """
        Initialize the facial expression classifier.
        
        Args:
            model_path: Path to pre-trained model file
            input_shape: Input shape expected by the model
            confidence_threshold: Minimum confidence for predictions
        """
        self.input_shape = input_shape
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.use_data_augmentation = True
        
        if model_path and os.path.isfile(model_path):
            # Load existing model
            try:
                self.model = load_model(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = self._build_model()
                print("Created new model (untrained)")
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
        Build an improved CNN model architecture for facial expression recognition.
        
        Returns:
            Compiled model
        """
        model = Sequential()
        
        # First Convolutional Block - more filters for better feature extraction
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', 
                        input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second Convolutional Block
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third Convolutional Block
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
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
        
        # Compile the model with a lower learning rate for better convergence
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
             batch_size=32, epochs=50, model_save_path=None):
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
        
        # Learning rate reduction on plateau
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Use data augmentation for better generalization
        if self.use_data_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            # Train with data augmentation
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without data augmentation
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def predict(self, face_image, apply_correction=True):
        """
        Predict expression for a face image with confidence filtering and bias correction.
        
        Args:
            face_image: Input face image
            apply_correction: Whether to apply bias correction for more accurate predictions
            
        Returns:
            Tuple of (expression_label, probabilities_dict)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Load or train a model first.")
        
        # Preprocess the face image
        processed_image = self._preprocess_face(face_image)
        
        # Make prediction
        prediction = self.model.predict(processed_image, verbose=0)[0]
        
        # Apply corrections to improve accuracy
        if apply_correction:
            # Correct for common biases in facial expression recognition
            # - Many models are biased towards 'angry' for certain ethnicities
            # - People with glasses are often misclassified as 'disgust'
            # - Subtle smiles are often not detected correctly
            
            # Check for glasses (simple heuristic based on edge detection)
            has_glasses = self._detect_glasses(face_image)
            
            if has_glasses and prediction[1] > 0.3:  # If high "disgust" prediction with glasses
                # Reduce disgust probability
                prediction[1] *= 0.7
                
                # Increase happy probability if the mouth appears to be smiling
                is_smiling = self._detect_smile(face_image)
                if is_smiling:
                    prediction[3] = max(prediction[3] * 1.5, prediction[3] + 0.2)
            
            # Normalize probabilities after correction
            prediction = prediction / np.sum(prediction)
        
        # Get the most likely expression
        expression_idx = np.argmax(prediction)
        expression = EXPRESSION_LABELS[expression_idx]
        
        # Apply confidence threshold
        max_prob = prediction[expression_idx]
        if max_prob < self.confidence_threshold:
            expression = "uncertain"
        
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
        
        # Apply histogram equalization for better contrast
        equalized_face = cv2.equalizeHist(resized_face)
        
        # Normalize pixel values to [0, 1]
        normalized_face = equalized_face / 255.0
        
        # Reshape to model input shape with batch dimension
        input_face = normalized_face.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        
        return input_face
    
    def _detect_glasses(self, face_image):
        """Simple heuristic to detect glasses on a face."""
        # Convert to grayscale
        if len(face_image.shape) == 3 and face_image.shape[2] > 1:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Get face height and width
        h, w = gray.shape[:2]
        
        # Define eye region (approximately)
        eye_region = gray[int(h*0.2):int(h*0.5), :]
        
        # Apply edge detection
        edges = cv2.Canny(eye_region, 100, 200)
        
        # Count edges in the eye region
        edge_count = np.count_nonzero(edges)
        
        # Heuristic: If there are many edges in the eye region, glasses might be present
        return edge_count > (eye_region.size * 0.05)
    
    def _detect_smile(self, face_image):
        """Simple heuristic to detect a smile on a face."""
        # Convert to grayscale
        if len(face_image.shape) == 3 and face_image.shape[2] > 1:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Get face height and width
        h, w = gray.shape[:2]
        
        # Define mouth region (approximately)
        mouth_region = gray[int(h*0.6):int(h*0.9), int(w*0.25):int(w*0.75)]
        
        # Apply edge detection
        edges = cv2.Canny(mouth_region, 100, 200)
        
        # Count edges in the mouth region
        edge_count = np.count_nonzero(edges)
        
        # Check for mouth curvature (simple approximation)
        if mouth_region.size > 0:
            # Divide mouth into left and right halves
            left_half = mouth_region[:, :mouth_region.shape[1]//2]
            right_half = mouth_region[:, mouth_region.shape[1]//2:]
            
            # Compute average intensities in upper and lower parts of each half
            upper_left = np.mean(left_half[:left_half.shape[0]//2])
            lower_left = np.mean(left_half[left_half.shape[0]//2:])
            upper_right = np.mean(right_half[:right_half.shape[0]//2])
            lower_right = np.mean(right_half[right_half.shape[0]//2:])
            
            # If lower parts are brighter than upper parts, this might indicate a smile
            smile_indicator = (lower_left > upper_left and lower_right > upper_right)
            
            return smile_indicator or edge_count > (mouth_region.size * 0.1)
        
        return False
    
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
