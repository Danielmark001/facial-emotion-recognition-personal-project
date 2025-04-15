#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved real-time facial expression recognition using webcam.
This version uses the enhanced classifier for better accuracy.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import time
from utils.face_detector import FaceDetector
from utils.visualization import draw_results
from models.improved_classifier import ImprovedExpressionClassifier

def main():
    """Main function for real-time facial expression recognition."""
    print("Starting Improved Real-time Facial Expression Recognition")
    print("Press 'q' to quit")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Falling back to image-based demo...")
        # Call image-based demo here
        return
    
    # Initialize face detector with Haar cascade method (more reliable than DNN for this demo)
    print("Initializing face detector...")
    face_detector = FaceDetector(confidence_threshold=0.5, method="haar")
    
    # Initialize improved expression classifier
    print("Initializing improved expression model...")
    model_path = os.path.join("models", "trained", "expression_model.keras")
    expression_model = ImprovedExpressionClassifier(model_path, confidence_threshold=0.4)
    
    # Performance tracking
    frame_count = 0
    fps_list = []
    start_time = time.time()
    
    # For stable prediction (to reduce flickering)
    smoothing_factor = 0.3
    prev_probs = None
    prev_expressions = []  # Keep track of recent expressions for temporal consistency
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            
            # Process the frame
            frame_start_time = time.time()
            
            # Detect faces
            faces = face_detector.detect(frame)
            
            # Process each detected face
            results = []
            for face_box in faces:
                # Extract face ROI
                x, y, w, h = face_box
                face_roi = frame[y:y+h, x:x+w]
                
                # Skip if face ROI is invalid
                if face_roi.size == 0:
                    continue
                
                # Classify expression
                expression, probabilities = expression_model.predict(face_roi, apply_correction=True)
                
                # Apply smoothing to reduce flickering
                if prev_probs is not None:
                    for emotion in probabilities:
                        # Weighted average with previous probabilities
                        probabilities[emotion] = smoothing_factor * probabilities[emotion] + \
                                               (1 - smoothing_factor) * prev_probs.get(emotion, 0)
                    
                    # Renormalize
                    total = sum(probabilities.values())
                    probabilities = {e: p/total for e, p in probabilities.items()}
                    
                    # Update expression based on smoothed probabilities
                    if expression != "uncertain":
                        new_expression = max(probabilities, key=probabilities.get)
                        
                        # Add to recent expressions list for temporal consistency
                        prev_expressions.append(new_expression)
                        if len(prev_expressions) > 5:  # Keep last 5 expressions
                            prev_expressions.pop(0)
                        
                        # Use majority vote for final expression
                        from collections import Counter
                        expression_counts = Counter(prev_expressions)
                        expression = expression_counts.most_common(1)[0][0]
                
                prev_probs = probabilities.copy()
                
                # Store results
                results.append({
                    'box': face_box,
                    'expression': expression,
                    'probabilities': probabilities
                })
            
            # Draw results on the frame
            output_frame = draw_results(frame, results)
            
            # Calculate FPS
            frame_time = time.time() - frame_start_time
            fps = 1 / max(frame_time, 0.001)  # Avoid division by zero
            fps_list.append(fps)
            if len(fps_list) > 30:  # Average over last 30 frames
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            
            # Update frame count
            frame_count += 1
            
            # Add performance info
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(output_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
            # Add face count
            face_text = f"Faces: {len(results)}"
            cv2.putText(output_frame, face_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("Facial Expression Recognition", output_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        total_time = time.time() - start_time
        print("\nPerformance Summary:")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count / total_time:.2f}")
        print("Demo ended")


if __name__ == "__main__":
    main()
