import cv2
import numpy as np

def detect_player(frame):
    """
    Detect a player in the given frame using a simple color-based approach.
    Assume the player is wearing a bright red outfit.
    
    :param frame: numpy array of shape (height, width, 3) representing an RGB image
    :return: tuple (x, y) representing the center of the detected player, or None if not found
    """
    # TODO: Implement player detection
    # Hint: You can use cv2.inRange() to create a mask for red color
    # Then find the largest contour in the mask
    pass

# Example usage (you don't need to implement this part in the interview)
# frame = cv2.imread('game_screenshot.jpg')
# player_position = detect_player(frame)
# if player_position:
#     print(f"Player detected at {player_position}")
# else:
#     print("Player not found")