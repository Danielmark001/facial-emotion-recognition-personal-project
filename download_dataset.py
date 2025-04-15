#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to download and prepare the FER2013 dataset.
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm

# URLs for dataset (alternative sources if Kaggle is not accessible)
FER_DATASET_URLS = [
    "https://www.kaggle.com/datasets/msambare/fer2013/download",  # Kaggle (requires authentication)
    "https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.csv",      # Dropbox link
    "https://drive.google.com/uc?export=download&id=1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr" # Google Drive link
]

# Create a progress bar for downloads
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    """Download a file with progress bar."""
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                              desc=os.path.basename(output_path)) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def download_from_alternative_sources(output_path):
    """Try downloading from alternative sources."""
    print("Trying alternative download sources...")
    
    for i, url in enumerate(FER_DATASET_URLS[1:], 1):  # Skip Kaggle
        print(f"Trying source {i}...")
        if download_file(url, output_path):
            print(f"Successfully downloaded from alternative source {i}.")
            return True
    
    return False

def create_sample_dataset(csv_path, output_path, sample_size=5000):
    """Create a smaller sample dataset for quick testing."""
    print(f"Creating sample dataset with {sample_size} examples...")
    
    # Read the original dataset
    df = pd.read_csv(csv_path)
    
    # Create stratified sample
    sample_df = pd.DataFrame()
    for emotion in df['emotion'].unique():
        emotion_df = df[df['emotion'] == emotion]
        # Calculate proportion of each class in the original dataset
        proportion = len(emotion_df) / len(df)
        # Sample based on proportion
        class_sample_size = max(int(sample_size * proportion), 100)  # At least 100 examples per class
        emotion_sample = emotion_df.sample(min(class_sample_size, len(emotion_df)), random_state=42)
        sample_df = pd.concat([sample_df, emotion_sample])
    
    # Shuffle the sample
    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    sample_df.to_csv(output_path, index=False)
    print(f"Sample dataset created and saved to {output_path}")
    
    return output_path

def main():
    """Main function to download and prepare the dataset."""
    # Setup paths
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    fer_csv_path = os.path.join(data_dir, "fer2013.csv")
    fer_sample_path = os.path.join(data_dir, "fer2013_sample.csv")
    
    # Check if dataset already exists
    if os.path.isfile(fer_csv_path):
        print(f"Dataset already exists at {fer_csv_path}")
    else:
        print("FER2013 dataset not found. Attempting to download...")
        print("\nNOTE: Kaggle requires authentication. If the download fails, please manually download")
        print("the dataset from https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"and place the 'fer2013.csv' file in the '{data_dir}' directory.")
        
        # Try alternative sources since Kaggle requires authentication
        if not download_from_alternative_sources(fer_csv_path):
            print("\nAutomatic download failed. Please manually download the dataset.")
            print("1. Go to https://www.kaggle.com/datasets/msambare/fer2013")
            print("2. Sign in to your Kaggle account")
            print("3. Download the dataset")
            print(f"4. Place 'fer2013.csv' in the '{data_dir}' directory")
            return False
    
    # Create a sample dataset for quick testing if the full dataset exists
    if os.path.isfile(fer_csv_path) and not os.path.isfile(fer_sample_path):
        create_sample_dataset(fer_csv_path, fer_sample_path)
    
    # Basic validation of the dataset
    if os.path.isfile(fer_csv_path):
        try:
            df = pd.read_csv(fer_csv_path)
            print("\nDataset Summary:")
            print(f"Total samples: {len(df)}")
            print("Class distribution:")
            emotion_map = {
                0: 'angry',
                1: 'disgust',
                2: 'fear',
                3: 'happy',
                4: 'sad',
                5: 'surprise',
                6: 'neutral'
            }
            for emotion_idx, count in df['emotion'].value_counts().sort_index().items():
                emotion = emotion_map.get(emotion_idx, f"Unknown ({emotion_idx})")
                percentage = (count / len(df)) * 100
                print(f"  {emotion}: {count} samples ({percentage:.2f}%)")
            return True
        except Exception as e:
            print(f"Error validating dataset: {e}")
            return False
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
