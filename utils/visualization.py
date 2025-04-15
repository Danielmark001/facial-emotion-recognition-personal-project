#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for facial expression recognition system.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


# Color mapping for different expressions
EXPRESSION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 140, 255),  # Orange
    'fear': (0, 255, 255),     # Yellow
    'happy': (0, 255, 0),      # Green
    'sad': (255, 0, 0),        # Blue
    'surprise': (255, 0, 255), # Magenta
    'neutral': (255, 255, 255) # White
}

# Default color if expression not in mapping
DEFAULT_COLOR = (200, 200, 200)  # Gray


def draw_results(frame, results):
    """
    Improved visualization function that ensures probabilities are fully visible.
    
    Args:
        frame: Input image frame
        results: List of result dictionaries containing 'box', 'expression', and 'probabilities'
        
    Returns:
        Annotated frame
    """
    # Make a copy to avoid modifying the original
    output_frame = frame.copy()
    
    # Process each result
    for result in results:
        face_box = result['box']
        expression = result['expression']
        probabilities = result.get('probabilities', None)
        
        # Get color for the detected expression
        color = EXPRESSION_COLORS.get(expression.lower(), DEFAULT_COLOR)
        
        # Draw face bounding box
        x, y, w, h = face_box
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw expression label
        label = f"{expression}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_frame, (x, y - label_size[1] - 10), 
                     (x + label_size[0] + 10, y), color, -1)
        cv2.putText(output_frame, label, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Improved probability visualization - ensure it fits on screen
        if probabilities is not None and isinstance(probabilities, dict):
            # Determine chart position - fixed location for better readability
            chart_x = 10
            chart_y = 70  # Start below FPS counter
            
            # Draw a background for the probability display
            chart_width = 300
            chart_height = len(probabilities) * 25 + 30
            cv2.rectangle(output_frame, (chart_x, chart_y), 
                         (chart_x + chart_width, chart_y + chart_height), 
                         (240, 240, 240), -1)
            
            # Add title
            cv2.putText(output_frame, "Expression Probabilities", 
                       (chart_x + 10, chart_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Sort probabilities by value
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            # Draw probability bars
            for i, (emotion, prob) in enumerate(sorted_probs):
                # Text position
                text_y = chart_y + 55 + i * 25
                
                # Draw emotion label
                emo_color = EXPRESSION_COLORS.get(emotion.lower(), DEFAULT_COLOR)
                cv2.putText(output_frame, f"{emotion}", 
                           (chart_x + 10, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Draw probability bar
                bar_length = int(prob * 150)  # Scale to 150 pixels
                cv2.rectangle(output_frame, 
                             (chart_x + 100, text_y - 15), 
                             (chart_x + 100 + bar_length, text_y - 5), 
                             emo_color, -1)
                
                # Draw probability value
                cv2.putText(output_frame, f"{prob:.2f}", 
                           (chart_x + 260, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return output_frame


def create_bar_chart(probabilities, width=150, height=100):
    """
    Create a bar chart image from expression probabilities.
    
    Args:
        probabilities: Dictionary of expression probabilities
        width: Width of the chart
        height: Height of the chart
        
    Returns:
        Bar chart as an image
    """
    # Sort probabilities by value in descending order
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 5 expressions
    top_expressions = sorted_probs[:5]
    
    # Extract expressions and probabilities
    expressions = [item[0] for item in top_expressions]
    probs = [item[1] for item in top_expressions]
    
    # Create figure and axis
    fig = Figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create horizontal bar chart
    bars = ax.barh(expressions, probs, color=[rgb_to_mpl(EXPRESSION_COLORS.get(expr.lower(), DEFAULT_COLOR)) 
                                           for expr in expressions])
    
    # Add probability values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f"{width:.2f}", va='center', fontsize=8)
    
    # Set labels and title
    ax.set_title('Expression Probabilities', fontsize=10)
    ax.set_xlim(0, 1)  # Probabilities are between 0 and 1
    
    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=8)
    
    # Render figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Convert to numpy array
    chart_img = np.array(canvas.renderer.buffer_rgba())
    
    # Convert from RGBA to BGR for OpenCV
    chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGBA2BGR)
    
    return chart_img


def rgb_to_mpl(bgr_color):
    """
    Convert OpenCV BGR color to matplotlib RGB color.
    
    Args:
        bgr_color: BGR color tuple (B, G, R)
        
    Returns:
        RGB color tuple normalized to range [0, 1]
    """
    b, g, r = bgr_color
    return (r / 255.0, g / 255.0, b / 255.0)


def overlay_image(background, overlay, x, y):
    """
    Overlay an image on top of another with alpha blending.
    
    Args:
        background: Background image
        overlay: Image to overlay
        x: X-coordinate to place overlay
        y: Y-coordinate to place overlay
    """
    h, w = overlay.shape[:2]
    
    # Ensure coordinates are within background image
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return
    
    # Check if overlay has alpha channel
    if overlay.shape[2] == 4:
        # Alpha blending
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.dstack([alpha, alpha, alpha])
        background_part = background[y:y+h, x:x+w]
        overlay_color = overlay[:, :, :3]
        
        # Apply alpha blending
        background[y:y+h, x:x+w] = (
            alpha * overlay_color + (1 - alpha) * background_part
        ).astype(np.uint8)
    else:
        # Simple overlay without alpha blending
        background[y:y+h, x:x+w] = overlay
