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
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2. Compute the average pixel intensity
    average_intensity = np.mean(gray_image)
    # 3. Detect edges using Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    # 4. Compute the number of detected edges
    edge_count = np.sum(edges > 0)
    # 5. Return a dictionary with these features
    
    features = {
        'average_intensity': average_intensity,
        'edge_count': edge_count,
    }
    return features

# Example usage (not needed in the interview):
# image = cv2.imread('game_character.jpg')
# features = extract_features(image)
# print(features)