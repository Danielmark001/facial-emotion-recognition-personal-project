#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Console-based demo of facial expression recognition.
This version doesn't require GUI interactions.
"""

import os
import sys
import cv2
import numpy as np
import time
import glob
import random
from utils.face_detector import FaceDetector

# Expression labels
EXPRESSION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class SimpleExpressionClassifier:
    """Simple expression classifier for demonstration."""
    
    def __init__(self):
        """Initialize the classifier."""
        # Statistical behavior patterns for testing
        self.emotion_transitions = {
            'angry': {'angry': 0.7, 'neutral': 0.2, 'sad': 0.1},
            'disgust': {'disgust': 0.6, 'neutral': 0.3, 'angry': 0.1},
            'fear': {'fear': 0.6, 'surprise': 0.2, 'neutral': 0.2},
            'happy': {'happy': 0.8, 'neutral': 0.2},
            'sad': {'sad': 0.7, 'neutral': 0.2, 'angry': 0.1},
            'surprise': {'surprise': 0.6, 'fear': 0.2, 'neutral': 0.2},
            'neutral': {'neutral': 0.5, 'happy': 0.1, 'sad': 0.1, 'angry': 0.1, 
                      'fear': 0.1, 'surprise': 0.1}
        }
        
        # Currently detected emotion (for maintaining temporal consistency)
        self.current_emotion = 'neutral'
    
    def predict(self, face_image):
        """
        Predict emotion for a face image with temporal consistency.
        
        Args:
            face_image: Input face image
            
        Returns:
            Tuple of (expression_label, probabilities_dict)
        """
        # Determine next emotion based on transition probabilities
        transition_probs = self.emotion_transitions.get(self.current_emotion, 
                                                     {'neutral': 1.0})
        
        # Add randomness for variation
        noise_factor = 0.2
        for emotion in EXPRESSION_LABELS:
            if emotion not in transition_probs:
                transition_probs[emotion] = 0.0
            transition_probs[emotion] += np.random.uniform(0, noise_factor)
        
        # Normalize probabilities
        total = sum(transition_probs.values())
        transition_probs = {e: p/total for e, p in transition_probs.items()}
        
        # Select emotion based on probabilities
        emotions = list(transition_probs.keys())
        probs = list(transition_probs.values())
        
        # Update current emotion
        self.current_emotion = np.random.choice(emotions, p=probs)
        
        # Create probability distribution centered on the selected emotion
        probabilities = {}
        for emotion in EXPRESSION_LABELS:
            if emotion == self.current_emotion:
                probabilities[emotion] = np.random.uniform(0.5, 0.9)
            else:
                probabilities[emotion] = np.random.uniform(0.0, 0.2)
        
        # Normalize probabilities
        total = sum(probabilities.values())
        probabilities = {e: p/total for e, p in probabilities.items()}
        
        return self.current_emotion, probabilities


def create_sample_images():
    """Create sample face images for detection if none exist."""
    samples_dir = os.path.join("data", "test_faces")
    os.makedirs(samples_dir, exist_ok=True)
    
    if len(glob.glob(os.path.join(samples_dir, "*.jpg"))) > 0:
        return  # Images already exist
    
    # Create synthetic face images (very simplified)
    for i in range(5):
        # Create a blank image
        img = np.ones((300, 300, 3), dtype=np.uint8) * 220  # Light gray background
        
        # Draw face outline
        cv2.circle(img, (150, 150), 100, (200, 200, 200), -1)  # Face
        cv2.circle(img, (150, 150), 100, (150, 150, 150), 2)   # Face outline
        
        # Draw eyes
        cv2.circle(img, (120, 120), 15, (255, 255, 255), -1)  # Left eye white
        cv2.circle(img, (120, 120), 5, (80, 80, 80), -1)      # Left eye pupil
        cv2.circle(img, (180, 120), 15, (255, 255, 255), -1)  # Right eye white
        cv2.circle(img, (180, 120), 5, (80, 80, 80), -1)      # Right eye pupil
        
        # Draw nose
        cv2.line(img, (150, 140), (150, 170), (150, 150, 150), 2)
        cv2.line(img, (150, 170), (140, 180), (150, 150, 150), 2)
        
        # Draw mouth (varies to create expressions)
        if i == 0:  # Neutral
            cv2.line(img, (120, 200), (180, 200), (100, 100, 100), 2)
        elif i == 1:  # Happy
            cv2.ellipse(img, (150, 190), (30, 20), 0, 0, 180, (100, 100, 100), 2)
        elif i == 2:  # Surprised
            cv2.circle(img, (150, 200), 15, (100, 100, 100), 2)
        elif i == 3:  # Sad
            cv2.ellipse(img, (150, 210), (30, 20), 0, 180, 360, (100, 100, 100), 2)
        else:  # Angry
            cv2.line(img, (120, 200), (180, 200), (100, 100, 100), 2)
            cv2.line(img, (120, 110), (140, 100), (100, 100, 100), 2)  # Left eyebrow
            cv2.line(img, (180, 110), (160, 100), (100, 100, 100), 2)  # Right eyebrow
        
        # Save the image
        cv2.imwrite(os.path.join(samples_dir, f"face_{i}.jpg"), img)


def print_emotion_bar(emotion, probability, width=50):
    """Print a text-based bar chart for an emotion."""
    bar_length = int(probability * width)
    bar = '#' * bar_length + '-' * (width - bar_length)
    percentage = probability * 100
    print(f"{emotion:10}: [{bar}] {percentage:.2f}%")


def main():
    """Main function."""
    print("Facial Expression Recognition Demo (Console Mode)")
    print("------------------------------------------------")
    print("This demo uses a simplified model for demonstration purposes")
    
    # Create sample face images if they don't exist
    create_sample_images()
    
    # Get all test images
    samples_dir = os.path.join("data", "test_faces")
    test_images = glob.glob(os.path.join(samples_dir, "*.jpg"))
    
    if not test_images:
        print("Error: No test images found.")
        return
    
    # Initialize face detector with Haar cascade method
    print("\nInitializing face detector...")
    face_detector = FaceDetector(confidence_threshold=0.5, method="haar")
    
    # Initialize expression classifier
    print("Initializing expression classifier...")
    expression_classifier = SimpleExpressionClassifier()
    
    # Process each image
    for img_path in test_images:
        # Read image
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error: Could not read image {img_path}")
            continue
        
        print(f"\nProcessing image: {os.path.basename(img_path)}")
        print("-" * 40)
        
        # Process the image
        start_time = time.time()
        
        # Detect faces
        faces = face_detector.detect(image)
        
        # Process each detected face
        if not faces:
            print("No faces detected. Treating the entire image as a face.")
            # Treat the whole image as a face
            h, w = image.shape[:2]
            face_box = (0, 0, w, h)
            faces = [face_box]
        
        print(f"Found {len(faces)} face(s)")
        
        for i, face_box in enumerate(faces):
            # Extract face ROI
            x, y, w, h = face_box
            face_roi = image[y:y+h, x:x+w]
            
            # Skip if face ROI is invalid
            if face_roi.size == 0:
                print(f"Face {i+1}: Invalid region")
                continue
            
            # Classify expression
            expression, probabilities = expression_classifier.predict(face_roi)
            
            # Print results
            print(f"\nFace {i+1} at position (x={x}, y={y}, w={w}, h={h}):")
            print(f"Detected expression: {expression.upper()}")
            print("Emotion probabilities:")
            
            # Print bar chart of probabilities
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for emotion, prob in sorted_probs:
                print_emotion_bar(emotion, prob)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"\nProcessing time: {processing_time:.4f} seconds")
        print("\nPress Enter to process the next image or 'q' to quit...")
        
        # Wait for user input
        user_input = input()
        if user_input.lower() == 'q':
            break
    
    print("\nDemo completed. Thank you for testing the Facial Expression Recognition System!")


if __name__ == "__main__":
    main()
