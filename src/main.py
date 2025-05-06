import cv2
import numpy as np

def process_omr(filename):
    # Load the image
    img = cv2.imread(filename)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to get a black-and-white image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours (which can be used to detect the filled bubbles)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize the results dictionary (this is where we store the answer key)
    results = {}
    
    # Example: Assuming you have 20 questions and bubbles are in fixed positions
    # You will need to adjust this for the actual OMR sheet structure you're working with
    question_positions = {
        "Q1": (100, 100),   # Example positions (x, y) of bubbles
        "Q2": (200, 100),
        "Q3": (300, 100),
        "Q4": (100, 200),
        # Add more questions based on your OMR sheet structure
    }

    # Iterate over each question and try to match it with the contours
    for question, pos in question_positions.items():
        x, y = pos
        # Extract the region of interest (ROI) where the bubbles are located
        roi = thresh[y:y+50, x:x+50]  # Adjust size based on your bubble area
        filled_bubble = detect_filled_bubble(roi)  # Custom function to check if bubble is filled
        
        # Assuming you have a way to map each bubble to A, B, C, D answers
        if filled_bubble == "A":
            results[question] = "A"
        elif filled_bubble == "B":
            results[question] = "B"
        elif filled_bubble == "C":
            results[question] = "C"
        else:
            results[question] = "D"

    return results


def detect_filled_bubble(roi):
    """Detect if the bubble is filled or not based on pixel intensity in the ROI."""
    # Find contours of the bubbles in the ROI (region of interest)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any of the contours match the expected bubble size
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # You might need to adjust this value for your bubbles
            return "A"  # Example logic, can be adjusted for real bubble answers
    return None  # No filled bubble detected
