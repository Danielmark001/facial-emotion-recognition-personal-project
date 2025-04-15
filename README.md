# Real-time Facial Expression Recognition System

A comprehensive computer vision project that implements real-time facial expression recognition using deep learning techniques. This system can detect faces in a video stream and classify facial expressions into seven emotional categories (happy, sad, angry, surprised, neutral, disgust, fear).


## Features

- **Real-time face detection** using OpenCV's Haar cascade or DNN methods
- **Facial expression classification** using a CNN model
- **Improved visualization** with clear emotion probability displays
- **Bias correction** for better accuracy with people wearing glasses
- **Multiple demo modes**:
  - Real-time webcam processing
  - Image-based processing
  - Headless processing for environments without GUI
- **Smoothing algorithms** to reduce prediction flickering

## Project Structure

```
facial_expression_recognition/
├── main.py                      # Original main application
├── real_time_demo.py            # Real-time demo with webcam
├── improved_real_time_demo.py   # Enhanced version with better accuracy
├── headless_demo.py             # Demo for environments without GUI
├── train.py                     # Script for training the model
├── generate_dataset.py          # Script to generate synthetic dataset
├── README.md                    # Project documentation
├── models/
│   ├── expression_classifier.py    # Original CNN model
│   ├── improved_classifier.py      # Enhanced model with bias correction
│   └── trained/                    # Pre-trained model weights
├── utils/
│   ├── face_detector.py         # Face detection module
│   ├── visualization.py         # Results visualization
│   └── dataset_utils.py         # Dataset handling utilities
└── data/
    ├── test_faces/              # Test images
    ├── sample_images/           # Sample emotion images
    └── output_frames/           # Frames produced by headless demo
```

## Setup and Installation

### Prerequisites

- Python 3.7+
- OpenCV
- TensorFlow 2.x
- NumPy
- Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/Danielmark001/facial-emotion-recognition-personal-project.git
cd facial-emotion-recognition-personal-project

# Install dependencies
pip install -r requirements.txt
```

### Generating a Model

```bash
# Generate a simple model for demonstration
python simple_gen.py

# Alternatively, train a model on the synthetic dataset
python generate_dataset.py  # Create synthetic dataset first
python train.py --data data/fer2013_synthetic.csv --model models/trained/expression_model.keras --epochs 10
```

## Usage

### Real-time Demo (Webcam)

```bash
# Run the original demo
python real_time_demo.py

# Run the improved version with better accuracy
python improved_real_time_demo.py
```

### Headless Demo (No GUI)

```bash
# Run the headless demo that saves frames to disk
python headless_demo.py
# Output frames will be saved in data/output_frames/
```

### Process Individual Images

```bash
# Run the image-based demo
python image_demo.py
```

## How It Works

### Face Detection

The system uses OpenCV's Haar cascade classifier for face detection, which is fast and reliable for frontal faces. It also supports OpenCV's DNN face detector for improved detection in challenging conditions.

### Expression Classification

The facial expression recognition uses a Convolutional Neural Network (CNN) with the following architecture:

1. Multiple convolutional blocks with batch normalization
2. Max pooling layers for feature reduction
3. Dropout layers to prevent overfitting
4. Dense layers for classification
5. Softmax output for 7 emotion categories

### Improved Classification

The improved classifier addresses common issues in facial expression recognition:

- **Glasses Detection**: Special handling for people wearing glasses, which can interfere with expression recognition
- **Bias Correction**: Adjustments for ethnic and cultural biases in expression detection
- **Smile Detection**: Enhanced detection of subtle smiles that are often misclassified
- **Temporal Consistency**: Tracking expressions over time to reduce flickering

### Visualization

The visualization component has been enhanced to clearly display:

- Face bounding boxes with the detected emotion
- A fixed-position probability chart that shows all emotions and their probabilities
- Clear, readable percentages that don't get cut off

## Training Your Own Model

For the best accuracy, you can train the model on the FER2013 dataset or your own dataset:

1. Obtain the FER2013 dataset from Kaggle
2. Place the CSV in the `data` directory
3. Run the training script:

```bash
python train.py --data data/fer2013.csv --model models/trained/custom_model.keras --epochs 50 --batch-size 32 --augment --plot
```

## Performance Optimization

For better performance on resource-constrained devices:

- Use the Haar cascade detector instead of DNN
- Reduce the input resolution
- Adjust the frame processing rate
- Use the simplified model architecture

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The FER2013 dataset for training facial expression models
- OpenCV for computer vision algorithms
- TensorFlow for the deep learning framework


