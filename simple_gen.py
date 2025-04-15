#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple model generator with minimal output.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Redirect stdout to null device to suppress TensorFlow output
orig_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

def create_model():
    # Create a minimal model
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Force model to initialize
    model.predict(np.zeros((1, 48, 48, 1)), verbose=0)
    
    return model

try:
    # Create model directory
    model_dir = os.path.join("models", "trained")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create and save model
    model = create_model()
    model_path = os.path.join(model_dir, "expression_model.keras")
    model.save(model_path, save_format='keras')
    
    # Restore stdout and print success message
    sys.stdout = orig_stdout
    print(f"Model created and saved to {model_path}")
    
except Exception as e:
    # Restore stdout and print error
    sys.stdout = orig_stdout
    print(f"Error: {e}")
