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
    range = cv2.inRange(frame, (0, 0, 200), (100, 100, 255))
    contours, _ = cv2.findContours(range, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

# Example usage (you don't need to implement this part in the interview)
# frame = cv2.imread('game_screenshot.jpg')
# player_position = detect_player(frame)
# if player_position:
#     print(f"Player detected at {player_position}")
# else:
#     print("Player not found")