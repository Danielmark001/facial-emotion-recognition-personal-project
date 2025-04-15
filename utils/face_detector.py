#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face detection module using OpenCV's DNN face detector
or dlib's HOG face detector as fallback.
"""

import os
import cv2
import numpy as np

class FaceDetector:
    """Face detector class supporting multiple face detection methods."""
    
    def __init__(self, confidence_threshold=0.5, method="dnn"):
        """
        Initialize face detector.
        
        Args:
            confidence_threshold: Minimum confidence for detection (for DNN method)
            method: Detection method ("dnn" or "hog")
        """
        self.confidence_threshold = confidence_threshold
        self.method = method
        
        if method == "dnn":
            # Initialize OpenCV DNN face detector
            # Path to the pre-trained model files
            self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         "models", "face_detector")
            
            # Create model paths - will download if they don't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            self.config_path = os.path.join(self.model_path, 
                                          "deploy.prototxt")
            self.weights_path = os.path.join(self.model_path, 
                                           "res10_300x300_ssd_iter_140000.caffemodel")
            
            # Download models if they don't exist
            self._ensure_models_exist()
            
            # Load the DNN model
            self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
            
        elif method == "hog":
            # Initialize dlib's HOG based face detector
            try:
                import dlib
                self.detector = dlib.get_frontal_face_detector()
            except ImportError:
                print("Error: dlib not installed. Please install dlib to use HOG method.")
                print("Falling back to OpenCV's Haar Cascade face detector.")
                self.method = "haar"
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                                    "haarcascade_frontalface_default.xml")
            except Exception as e:
                print(f"Error initializing dlib: {e}")
                print("Falling back to OpenCV's Haar Cascade face detector.")
                self.method = "haar"
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                                    "haarcascade_frontalface_default.xml")
        elif method == "haar":
            # OpenCV's Haar Cascade face detector
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                                "haarcascade_frontalface_default.xml")
        else:
            raise ValueError(f"Unsupported detection method: {method}")
    
    def _ensure_models_exist(self):
        """Ensure the face detection model files exist."""
        # Check if model files exist
        if not os.path.isfile(self.config_path) or not os.path.isfile(self.weights_path):
            print("Face detection model files not found. Downloading...")
            
            # URLs for the model files
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            
            # Download the files
            try:
                import urllib.request
                
                # Download the prototxt
                if not os.path.isfile(self.config_path):
                    print(f"Downloading {prototxt_url}...")
                    urllib.request.urlretrieve(prototxt_url, self.config_path)
                
                # Download the model weights
                if not os.path.isfile(self.weights_path):
                    print(f"Downloading {model_url}...")
                    urllib.request.urlretrieve(model_url, self.weights_path)
                
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model files: {e}")
                print("Please download the files manually:")
                print(f"1. {prototxt_url} -> {self.config_path}")
                print(f"2. {model_url} -> {self.weights_path}")
    
    def detect(self, image):
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        if self.method == "dnn":
            return self._detect_dnn(image)
        elif self.method == "hog":
            return self._detect_hog(image)
        elif self.method == "haar":
            return self._detect_haar(image)
    
    def _detect_dnn(self, image):
        """Detect faces using OpenCV's DNN face detector."""
        h, w = image.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), 
                                    [104, 117, 123], False, False)
        
        # Set the blob as input to the network
        self.net.setInput(blob)
        
        # Forward pass to get detections
        detections = self.net.forward()
        
        # Process detections
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter based on confidence
            if confidence > self.confidence_threshold:
                # Get box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                
                # Ensure box is within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Convert to (x, y, w, h) format
                faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def _detect_hog(self, image):
        """Detect faces using dlib's HOG face detector."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        dlib_rects = self.detector(gray, 1)
        
        # Convert dlib rectangles to (x, y, w, h) format
        faces = []
        for rect in dlib_rects:
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            faces.append((x, y, w, h))
        
        return faces
    
    def _detect_haar(self, image):
        """Detect faces using OpenCV's Haar Cascade face detector."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
