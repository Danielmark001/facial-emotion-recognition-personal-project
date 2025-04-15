#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script for facial expression recognition system.
Uses a webcam and placeholder emotions for demonstration.
"""

import cv2
import numpy as np
import time
import random
from utils.face_detector import FaceDetector
from utils.visualization import draw_results, EXPRESSION_COLORS

def main():
    """Main demo function."""
    print("Starting Facial Expression Recognition Demo")
    print("This is a demonstration with placeholder emotions")
    print("Press 'q' to quit")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize face detector with Haar cascade method
    print("Initializing face detector...")
    face_detector = FaceDetector(confidence_threshold=0.5, method="haar")
    
    # Sample emotions for demonstration (matching the ones in visualization.py's EXPRESSION_COLORS)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            
            # Process the frame
            start_time = time.time()
            
            # Detect faces
            faces = face_detector.detect(frame)
            
            # Process each detected face with placeholder emotions
            results = []
            for face_box in faces:
                # Generate random emotion for demo
                emotion = random.choice(emotions)
                
                # Generate random probabilities
                probs = {e: random.random() for e in emotions}
                # Normalize probabilities so they sum to 1
                total = sum(probs.values())
                probs = {e: p/total for e, p in probs.items()}
                
                # Make the selected emotion have highest probability
                max_prob = max(probs.values())
                for e in probs:
                    if e == emotion:
                        probs[e] = max_prob * 1.5
                
                # Re-normalize
                total = sum(probs.values())
                probs = {e: p/total for e, p in probs.items()}
                
                # Store results
                results.append({
                    'box': face_box,
                    'expression': emotion,
                    'probabilities': probs
                })
            
            # Draw results on the frame
            output_frame = draw_results(frame, results)
            
            # Calculate FPS
            processing_time = time.time() - start_time
            fps_text = f"FPS: {1/processing_time:.2f}"
            cv2.putText(output_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
            # Add demo mode notice
            demo_text = "DEMO MODE - Emotions are randomly generated"
            cv2.putText(output_frame, demo_text, (10, output_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("Facial Expression Recognition Demo", output_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Demo ended")


if __name__ == "__main__":
    main()
