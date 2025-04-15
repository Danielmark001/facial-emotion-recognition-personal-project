#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick training script to create a basic model from the synthetic dataset.
This is for demonstration purposes to have a working model quickly.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import cv2

# Expression labels
EXPRESSION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_synthetic_dataset(csv_path):
    """Load and process the synthetic dataset."""
    print(f"Loading dataset from {csv_path}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract pixel data and convert to numpy arrays
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    X = np.stack(pixels.values)
    
    # Reshape to 48x48 images with single channel
    X = X.reshape(-1, 48, 48, 1)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Extract labels
    y = df['emotion'].values
    
    # Convert to one-hot encoding
    y = to_categorical(y, num_classes=len(EXPRESSION_LABELS))
    
    return X, y

def create_simple_model(input_shape=(48, 48, 1)):
    """Create a simplified CNN model for facial expression recognition."""
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', 
                    input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(len(EXPRESSION_LABELS), activation='softmax'))
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def train_quick_model():
    """Train a quick model on synthetic data."""
    # Setup paths
    data_dir = "data"
    model_dir = os.path.join("models", "trained")
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Dataset path
    synthetic_dataset = os.path.join(data_dir, "fer2013_synthetic.csv")
    
    # Check if dataset exists
    if not os.path.isfile(synthetic_dataset):
        print("Synthetic dataset not found. Please run generate_dataset.py first.")
        return False
    
    # Load dataset
    X, y = load_synthetic_dataset(synthetic_dataset)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create model
    model = create_simple_model()
    model.summary()
    
    # Define callbacks
    model_path = os.path.join(model_dir, "expression_model.keras")
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model (with fewer epochs for quick training)
    epochs = 10
    batch_size = 32
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Save the model (in case checkpoint didn't save it)
    if not os.path.isfile(model_path):
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    return True

if __name__ == "__main__":
    print("Starting quick model training...")
    success = train_quick_model()
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed.")
