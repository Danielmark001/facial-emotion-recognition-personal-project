#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for the facial expression recognition model.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from models.expression_classifier import ExpressionClassifier


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train facial expression recognition model")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset file (CSV format)")
    parser.add_argument("--model", type=str, default="models/expression_model.h5",
                        help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--augment", action="store_true",
                        help="Use data augmentation")
    parser.add_argument("--plot", action="store_true",
                        help="Plot training history")
    return parser.parse_args()


def load_fer_dataset(csv_path):
    """
    Load the Facial Expression Recognition dataset.
    
    Expected CSV format:
    - emotion: Label (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    - pixels: Flattened 48x48 grayscale image pixel values
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        X: Image data (N, 48, 48, 1)
        y: One-hot encoded labels
    """
    print(f"Loading dataset from {csv_path}...")
    
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Extract pixel data and convert to numpy arrays
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    X = np.stack(pixels.values)
    
    # Reshape to 48x48 images
    X = X.reshape(-1, 48, 48, 1)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Extract labels
    y = df['emotion'].values
    
    # Convert to one-hot encoding
    y = to_categorical(y)
    
    return X, y


def create_data_generators(X_train, y_train, X_val, y_val, batch_size=64, augment=False):
    """
    Create data generators for training and validation.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        batch_size: Batch size
        augment: Whether to use data augmentation
        
    Returns:
        train_gen: Training data generator
        val_gen: Validation data generator
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    if augment:
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        # No augmentation
        train_datagen = ImageDataGenerator()
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
    
    return train_gen, val_gen


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history
        save_path: Path to save the plot image
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Check if dataset file exists
    if not os.path.isfile(args.data):
        print(f"Error: Dataset file '{args.data}' not found.")
        print(f"Checking for synthetic dataset...")
        synthetic_dataset = os.path.join(os.path.dirname(args.data), "fer2013_synthetic.csv")
        if os.path.isfile(synthetic_dataset):
            print(f"Found synthetic dataset at {synthetic_dataset}")
            args.data = synthetic_dataset
        else:
            print("No synthetic dataset found. Please run generate_dataset.py first.")
            return
    
    # Load dataset
    X, y = load_fer_dataset(args.data)
    
    # Print dataset information
    print(f"Dataset loaded: {X.shape[0]} samples, {y.shape[1]} classes")
    print(f"Image shape: {X.shape[1:]} (48x48 grayscale)")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create model
    input_shape = X_train.shape[1:]  # (48, 48, 1)
    model = ExpressionClassifier(model_path=None, input_shape=input_shape)
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    
    if args.augment:
        # Use data generators with augmentation
        print("Using data augmentation for training.")
        train_gen, val_gen = create_data_generators(
            X_train, y_train, X_val, y_val, 
            batch_size=args.batch_size, augment=True
        )
        
        # Train with generators
        history = model.model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            steps_per_epoch=len(X_train) // args.batch_size,
            validation_steps=len(X_val) // args.batch_size,
            verbose=1
        )
    else:
        # Train directly with arrays
        history = model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_save_path=args.model
        )
    
    # Save model if not already saved by the training process
    if not os.path.isfile(args.model):
        model.save(args.model)
    
    # Plot training history if requested
    if args.plot:
        plot_dir = os.path.join(os.path.dirname(args.model), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "training_history.png")
        plot_training_history(history, save_path=plot_path)


if __name__ == "__main__":
    main()
