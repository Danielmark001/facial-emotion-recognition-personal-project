#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to generate a synthetic FER2013-like dataset for testing.
This is an alternative when the actual dataset cannot be downloaded.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

def generate_synthetic_dataset(output_path, num_samples=5000):
    """
    Generate a synthetic dataset with similar structure to FER2013.
    
    Args:
        output_path: Path to save the CSV file
        num_samples: Number of samples to generate
        
    Returns:
        Path to the generated dataset
    """
    print(f"Generating synthetic dataset with {num_samples} examples...")
    
    # FER2013 uses 48x48 grayscale images
    img_size = 48
    
    # Create an empty DataFrame with the same structure as FER2013
    df = pd.DataFrame(columns=['emotion', 'pixels', 'Usage'])
    
    # Emotion categories (same as FER2013)
    emotions = {
        0: 'angry',
        1: 'disgust', 
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    # Usage categories
    usages = ['Training', 'PublicTest', 'PrivateTest']
    
    # Usage distribution
    usage_dist = {'Training': 0.8, 'PublicTest': 0.1, 'PrivateTest': 0.1}
    
    # Generate samples for each emotion
    samples_per_emotion = num_samples // len(emotions)
    
    for emotion_idx, emotion_name in emotions.items():
        print(f"Generating {samples_per_emotion} samples for emotion: {emotion_name}")
        
        for i in tqdm(range(samples_per_emotion)):
            # Create a random face-like pattern (just for demonstration)
            # In a real scenario, you would use actual face images
            img = np.zeros((img_size, img_size), dtype=np.uint8)
            
            # Add some random shapes that vaguely resemble facial features
            # Center of the face
            center_x, center_y = img_size // 2, img_size // 2
            
            # Face outline (circle)
            cv2.circle(img, (center_x, center_y), 
                     int(img_size * 0.4), 
                     255, 
                     thickness=np.random.randint(1, 3))
            
            # Left eye
            eye_size = np.random.randint(5, 8)
            left_eye_x = center_x - int(img_size * 0.15)
            left_eye_y = center_y - int(img_size * 0.1)
            cv2.circle(img, (left_eye_x, left_eye_y), 
                     eye_size, 
                     255, 
                     thickness=np.random.randint(1, 3))
            
            # Right eye
            right_eye_x = center_x + int(img_size * 0.15)
            right_eye_y = center_y - int(img_size * 0.1)
            cv2.circle(img, (right_eye_x, right_eye_y), 
                     eye_size, 
                     255, 
                     thickness=np.random.randint(1, 3))
            
            # Mouth (varies by emotion)
            mouth_y = center_y + int(img_size * 0.2)
            
            if emotion_idx == 3:  # Happy
                # Smile: curved upward line
                cv2.ellipse(img, 
                          (center_x, mouth_y),
                          (int(img_size * 0.2), int(img_size * 0.1)),
                          0, 0, 180, 255, 
                          thickness=np.random.randint(1, 3))
            elif emotion_idx == 4:  # Sad
                # Frown: curved downward line
                cv2.ellipse(img, 
                          (center_x, mouth_y - int(img_size * 0.1)),
                          (int(img_size * 0.2), int(img_size * 0.1)),
                          0, 180, 360, 255, 
                          thickness=np.random.randint(1, 3))
            elif emotion_idx == 5:  # Surprise
                # Open mouth: circle
                cv2.circle(img, (center_x, mouth_y), 
                         int(img_size * 0.1), 
                         255, 
                         thickness=np.random.randint(1, 3))
            else:  # Other emotions
                # Straight line
                cv2.line(img, 
                       (center_x - int(img_size * 0.15), mouth_y),
                       (center_x + int(img_size * 0.15), mouth_y),
                       255, 
                       thickness=np.random.randint(1, 3))
            
            # Add some random noise
            noise = np.random.randint(0, 50, size=(img_size, img_size), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            # Convert image to string format (space-separated pixel values)
            pixels_str = ' '.join(map(str, img.flatten()))
            
            # Determine usage based on distribution
            usage = np.random.choice(usages, p=[usage_dist[u] for u in usages])
            
            # Add to DataFrame
            df.loc[len(df)] = [emotion_idx, pixels_str, usage]
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved to {output_path}")
    
    return output_path

def main():
    """Main function."""
    # Setup paths
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    fer_csv_path = os.path.join(data_dir, "fer2013_synthetic.csv")
    
    # Generate dataset
    generate_synthetic_dataset(fer_csv_path, num_samples=3500)  # Smaller for quick testing
    
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
            
            # Create example images to verify
            images_dir = os.path.join(data_dir, "sample_images")
            os.makedirs(images_dir, exist_ok=True)
            
            print("\nSaving sample images to verify...")
            for emotion_idx in range(7):
                emotion_samples = df[df['emotion'] == emotion_idx].head(5)
                for i, (_, row) in enumerate(emotion_samples.iterrows()):
                    pixels = np.array([int(p) for p in row['pixels'].split()]).reshape((48, 48))
                    emotion_name = emotion_map.get(emotion_idx)
                    img_path = os.path.join(images_dir, f"{emotion_name}_{i}.png")
                    cv2.imwrite(img_path, pixels)
            
            print(f"Sample images saved to {images_dir}")
            return True
        except Exception as e:
            print(f"Error validating dataset: {e}")
            return False
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
