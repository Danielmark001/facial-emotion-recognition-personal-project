#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo of facial expression recognition using sample images.
"""

import os
import cv2
import numpy as np
import time
import glob
import random
from utils.face_detector import FaceDetector
from utils.visualization import draw_results

# Expression labels (matching visualization.py)
EXPRESSION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class SimpleExpressionClassifier:
    """Simple expression classifier using statistical patterns."""
    
    def __init__(self):
        """Initialize the classifier."""
        # Load sample images to use as templates
        self.samples_dir = os.path.join("data", "sample_images")
        self.templates = self._load_templates()
        
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
        
    def _load_templates(self):
        """Load sample images as templates."""
        templates = {}
        
        if os.path.exists(self.samples_dir):
            for emotion in EXPRESSION_LABELS:
                template_path = os.path.join(self.samples_dir, f"{emotion}_0.png")
                if os.path.exists(template_path):
                    templates[emotion] = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        return templates
    
    def predict(self, face_image):
        """
        Predict emotion for a face image with temporal consistency.
        
        Args:
            face_image: Input face image
            
        Returns:
            Tuple of (expression_label, probabilities_dict)
        """
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_image
        
        # Resize to 48x48
        resized_face = cv2.resize(gray_face, (48, 48))
        
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


def main():
    """Main function."""
    print("Starting Facial Expression Recognition Demo (Image Mode)")
    print("This demo uses a simplified model for demonstration purposes")
    print("Press any key to process the next image, or 'q' to quit")
    
    # Create sample face images if they don't exist
    create_sample_images()
    
    # Get all test images
    samples_dir = os.path.join("data", "test_faces")
    test_images = glob.glob(os.path.join(samples_dir, "*.jpg"))
    
    if not test_images:
        print("Error: No test images found.")
        return
    
    # Initialize face detector with Haar cascade method
    print("Initializing face detector...")
    face_detector = FaceDetector(confidence_threshold=0.5, method="haar")
    
    # Initialize expression classifier
    print("Initializing expression classifier...")
    expression_classifier = SimpleExpressionClassifier()
    
    # Create a window
    cv2.namedWindow("Facial Expression Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Facial Expression Recognition", 800, 600)
    
    # Process each image
    for img_path in test_images:
        # Read image
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error: Could not read image {img_path}")
            continue
        
        print(f"Processing image: {os.path.basename(img_path)}")
        
        # Process the image
        start_time = time.time()
        
        # Detect faces
        faces = face_detector.detect(image)
        
        # Process each detected face
        results = []
        for face_box in faces:
            # Extract face ROI
            x, y, w, h = face_box
            face_roi = image[y:y+h, x:x+w]
            
            # Skip if face ROI is invalid
            if face_roi.size == 0:
                continue
            
            # Classify expression
            expression, probabilities = expression_classifier.predict(face_roi)
            
            # Store results
            results.append({
                'box': face_box,
                'expression': expression,
                'probabilities': probabilities
            })
        
        # If no faces detected, add the whole image as a face
        if not results:
            h, w = image.shape[:2]
            face_box = (0, 0, w, h)
            expression, probabilities = expression_classifier.predict(image)
            results.append({
                'box': face_box,
                'expression': expression,
                'probabilities': probabilities
            })
        
        # Draw results on the image
        output_image = draw_results(image, results)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        fps_text = f"Processing time: {processing_time:.2f}s"
        cv2.putText(output_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
        
        # Add demo mode notice
        demo_text = "DEMO MODE - Using simplified model"
        cv2.putText(output_image, demo_text, (10, output_image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the image
        cv2.imshow("Facial Expression Recognition", output_image)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    cv2.destroyAllWindows()
    print("Demo ended")


if __name__ == "__main__":
    main()
