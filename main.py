#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time Facial Expression Recognition System
Main application entry point
"""

import os
import argparse
import time
import cv2
import numpy as np
from utils.face_detector import FaceDetector
from utils.visualization import draw_results
from models.expression_classifier import ExpressionClassifier


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Facial Expression Recognition System")
    parser.add_argument("--source", type=str, default="webcam", 
                        choices=["webcam", "video", "image"],
                        help="Source type (webcam, video file, or image)")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to video or image file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output")
    parser.add_argument("--model", type=str, default="models/expression_model.h5",
                        help="Path to facial expression model")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Minimum confidence for face detection")
    parser.add_argument("--display", action="store_true",
                        help="Display the output")
    return parser.parse_args()


def process_frame(frame, face_detector, expression_classifier):
    """
    Process a single frame for face detection and expression recognition.
    
    Args:
        frame: Input image frame
        face_detector: Instance of FaceDetector
        expression_classifier: Instance of ExpressionClassifier
        
    Returns:
        Processed frame with annotations
    """
    # Make a copy to avoid modifying the original
    output_frame = frame.copy()
    
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
        expression, probabilities = expression_classifier.predict(face_roi)
        
        # Store results
        results.append({
            'box': face_box,
            'expression': expression,
            'probabilities': probabilities
        })
    
    # Draw results on the output frame
    annotated_frame = draw_results(output_frame, results)
    
    return annotated_frame


def run_webcam_mode(args, face_detector, expression_classifier):
    """Run the system in webcam mode."""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize output video writer if specified
    out = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            
            # Process the frame
            start_time = time.time()
            output_frame = process_frame(frame, face_detector, expression_classifier)
            processing_time = time.time() - start_time
            
            # Add FPS info
            fps_text = f"FPS: {1/processing_time:.2f}"
            cv2.putText(output_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
            # Write to output if specified
            if out:
                out.write(output_frame)
            
            # Display if requested
            if args.display:
                cv2.imshow("Facial Expression Recognition", output_frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        # Clean up
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


def run_video_mode(args, face_detector, expression_classifier):
    """Run the system on a video file."""
    # Check if file exists
    if not os.path.isfile(args.path):
        print(f"Error: Video file '{args.path}' not found.")
        return
    
    # Open video file
    cap = cv2.VideoCapture(args.path)
    
    # Check if opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.path}'.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize output video writer if specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"Processing video: {args.path}")
    print(f"Total frames: {total_frames}")
    print("Press 'q' to quit")
    
    frame_count = 0
    
    try:
        while True:
            # Read frame from video
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or error reading frame.")
                break
            
            frame_count += 1
            
            # Process the frame
            start_time = time.time()
            output_frame = process_frame(frame, face_detector, expression_classifier)
            processing_time = time.time() - start_time
            
            # Add FPS and progress info
            fps_text = f"FPS: {1/processing_time:.2f}"
            progress_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(output_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            cv2.putText(output_frame, progress_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
            # Write to output if specified
            if out:
                out.write(output_frame)
            
            # Display if requested
            if args.display:
                cv2.imshow("Facial Expression Recognition", output_frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        # Clean up
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


def run_image_mode(args, face_detector, expression_classifier):
    """Run the system on a single image."""
    # Check if file exists
    if not os.path.isfile(args.path):
        print(f"Error: Image file '{args.path}' not found.")
        return
    
    # Read the image
    image = cv2.imread(args.path)
    
    if image is None:
        print(f"Error: Could not read image file '{args.path}'.")
        return
    
    # Process the image
    print(f"Processing image: {args.path}")
    output_image = process_frame(image, face_detector, expression_classifier)
    
    # Save output if specified
    if args.output:
        cv2.imwrite(args.output, output_image)
        print(f"Output saved to: {args.output}")
    
    # Display if requested
    if args.display:
        cv2.imshow("Facial Expression Recognition", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize face detector (use haar method to avoid dlib dependency issues)
    print("Initializing face detector...")
    face_detector = FaceDetector(confidence_threshold=args.confidence, method="haar")
    
    # Initialize expression classifier
    print("Loading facial expression model...")
    expression_classifier = ExpressionClassifier(model_path=args.model)
    
    # Run the appropriate mode
    if args.source == "webcam":
        run_webcam_mode(args, face_detector, expression_classifier)
    elif args.source == "video":
        if not args.path:
            print("Error: Video path not specified.")
            return
        run_video_mode(args, face_detector, expression_classifier)
    elif args.source == "image":
        if not args.path:
            print("Error: Image path not specified.")
            return
        run_image_mode(args, face_detector, expression_classifier)


if __name__ == "__main__":
    main()
