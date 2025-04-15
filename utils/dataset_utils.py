#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for managing facial expression datasets.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm


def download_fer2013(data_dir="data"):
    """
    Download and extract the FER2013 dataset.
    
    Args:
        data_dir: Directory to save the dataset
        
    Returns:
        Path to the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # URL of the dataset
    # Note: This URL may change; update if necessary
    url = "https://www.kaggle.com/datasets/msambare/fer2013/download"
    
    csv_path = os.path.join(data_dir, "fer2013.csv")
    
    # Check if dataset already exists
    if os.path.isfile(csv_path):
        print(f"Dataset already exists at {csv_path}")
        return csv_path
    
    # Since Kaggle requires authentication, provide instructions for manual download
    print("FER2013 dataset needs to be manually downloaded from Kaggle:")
    print("1. Go to https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Sign in to your Kaggle account")
    print("3. Download the dataset and extract it")
    print(f"4. Place 'fer2013.csv' in the '{data_dir}' directory")
    
    return None


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update progress bar.
        
        Args:
            b: Number of blocks transferred
            bsize: Size of each block in bytes
            tsize: Total size in bytes
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        output_path: Path to save the file
        
    Returns:
        Path to the downloaded file
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                            miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                 reporthook=t.update_to)
    return output_path


def extract_zip(zip_path, extract_dir):
    """
    Extract a ZIP file.
    
    Args:
        zip_path: Path to the ZIP file
        extract_dir: Directory to extract to
        
    Returns:
        Path to the extracted directory
    """
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir


def create_train_val_test_split(csv_path, output_dir="data", 
                              val_ratio=0.15, test_ratio=0.15):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        csv_path: Path to the main CSV file
        output_dir: Directory to save the split files
        val_ratio: Ratio of validation set
        test_ratio: Ratio of test set
        
    Returns:
        Dictionary of paths to the split files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Get the total number of samples
    n_samples = len(df)
    
    # Calculate the number of samples for each split
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_test - n_val
    
    # Split the dataset
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train+n_val]
    df_test = df.iloc[n_train+n_val:]
    
    # Save the splits
    train_path = os.path.join(output_dir, "fer2013_train.csv")
    val_path = os.path.join(output_dir, "fer2013_val.csv")
    test_path = os.path.join(output_dir, "fer2013_test.csv")
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Training set saved to {train_path}: {len(df_train)} samples")
    print(f"Validation set saved to {val_path}: {len(df_val)} samples")
    print(f"Test set saved to {test_path}: {len(df_test)} samples")
    
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path
    }


def convert_csv_to_images(csv_path, output_dir, prefix=""):
    """
    Convert CSV pixel data to image files.
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save the images
        prefix: Prefix for the image filenames
        
    Returns:
        Dictionary mapping image paths to labels
    """
    import cv2
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Mapping of class indices to emotions
    emotion_map = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    # Create subdirectories for each emotion
    for emotion in emotion_map.values():
        os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)
    
    # Convert pixels to images
    print(f"Converting {len(df)} samples to images...")
    image_paths = {}
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Get emotion label
        emotion_idx = row['emotion']
        emotion = emotion_map[emotion_idx]
        
        # Convert pixel string to image
        pixels = np.array(row['pixels'].split(), dtype='float32')
        img = pixels.reshape(48, 48).astype('uint8')
        
        # Create filename
        filename = f"{prefix}_{i:06d}_{emotion}.png"
        img_path = os.path.join(output_dir, emotion, filename)
        
        # Save image
        cv2.imwrite(img_path, img)
        
        # Store image path and label
        image_paths[img_path] = emotion_idx
    
    print(f"Images saved to {output_dir}")
    
    return image_paths


def explore_dataset(csv_path):
    """
    Explore the dataset and print statistics.
    
    Args:
        csv_path: Path to the CSV file
    """
    # Read the CSV file
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Print basic information
    print(f"\nDataset overview:")
    print(f"Total samples: {len(df)}")
    
    # Print class distribution
    emotion_map = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    print("\nClass distribution:")
    class_counts = df['emotion'].value_counts().sort_index()
    
    for emotion_idx, count in class_counts.items():
        emotion = emotion_map.get(emotion_idx, f"Unknown ({emotion_idx})")
        percentage = (count / len(df)) * 100
        print(f"  {emotion}: {count} samples ({percentage:.2f}%)")
    
    # Print sample statistics
    print("\nPixel value statistics:")
    pixel_values = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    all_pixels = np.concatenate(pixel_values.values)
    
    print(f"  Min pixel value: {all_pixels.min()}")
    print(f"  Max pixel value: {all_pixels.max()}")
    print(f"  Mean pixel value: {all_pixels.mean():.2f}")
    print(f"  Standard deviation: {all_pixels.std():.2f}")
    
    return df


if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    
    # Download dataset
    csv_path = download_fer2013(data_dir)
    
    if csv_path and os.path.isfile(csv_path):
        # Explore dataset
        df = explore_dataset(csv_path)
        
        # Split dataset
        split_paths = create_train_val_test_split(csv_path, data_dir)
        
        # Convert to images (optional)
        # train_images = convert_csv_to_images(split_paths['train'], 
        #                                     os.path.join(data_dir, 'images', 'train'), 
        #                                     prefix='train')
