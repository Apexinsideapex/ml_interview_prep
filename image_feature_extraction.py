import numpy as np
import cv2

def extract_features(image):
    """
    Extract basic features from an input image.
    
    :param image: numpy array of shape (height, width, 3) representing an RGB image
    :return: dict containing extracted features
    """
    # TODO: Implement feature extraction
    # 1. Convert the image to grayscale
    # 2. Compute the average pixel intensity
    # 3. Detect edges using Canny edge detection
    # 4. Compute the number of detected edges
    # 5. Return a dictionary with these features
    
    features = {
        'average_intensity': 0,  # Replace with actual value
        'edge_count': 0,  # Replace with actual value
    }
    return features

# Example usage (not needed in the interview):
# image = cv2.imread('game_character.jpg')
# features = extract_features(image)
# print(features)