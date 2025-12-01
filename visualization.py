# File: visualization.py
# Description: Optimized for Real-Time speed using OpenCV.
# Updated: Thicker lines and larger joints for better visibility.

import cv2
import numpy as np

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Pairs of keypoints that form the skeleton lines.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    # --- HEAD TO SHOULDERS ---
    (0, 5): (255, 0, 255), # Head -> Left Shoulder
    (0, 6): (0, 255, 255), # Head -> Right Shoulder
    
    # --- TORSO ---
    (5, 6): (255, 255, 0), # Shoulder to Shoulder
    (5, 11): (255, 0, 255), # Left Shoulder -> Hip
    (6, 12): (0, 255, 255), # Right Shoulder -> Hip
    (11, 12): (255, 255, 0), # Hip to Hip
    
    # --- ARMS ---
    (5, 7): (255, 0, 255), # Left Shoulder -> Elbow
    (7, 9): (255, 0, 255), # Left Elbow -> Wrist
    (6, 8): (0, 255, 255), # Right Shoulder -> Elbow
    (8, 10): (0, 255, 255), # Right Elbow -> Wrist
    
    # --- LEGS ---
    (11, 13): (255, 0, 255), # Left Hip -> Knee
    (13, 15): (255, 0, 255), # Left Knee -> Ankle
    (12, 14): (0, 255, 255), # Right Hip -> Knee
    (14, 16): (0, 255, 255)  # Right Knee -> Ankle
}

def draw_prediction_on_image(image, keypoints_with_scores):
    """
    Draws the keypoint predictions on image using OpenCV.
    """
    height, width, channel = image.shape
    output_image = image.copy()
    
    keypoints = np.squeeze(keypoints_with_scores)
    score_threshold = 0.1

    # 1. Draw Edges (Limbs)
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        start_idx, end_idx = edge_pair
        start_kpt = keypoints[start_idx]
        end_kpt = keypoints[end_idx]
        
        if start_kpt[2] > score_threshold and end_kpt[2] > score_threshold:
            start_y, start_x = start_kpt[0], start_kpt[1]
            end_y, end_x = end_kpt[0], end_kpt[1]
            
            start_point = (int(start_x * width), int(start_y * height))
            end_point = (int(end_x * width), int(end_y * height))
            
            # INCREASED THICKNESS: 4 (was 3)
            cv2.line(output_image, start_point, end_point, color, 4)

    # 2. Draw Keypoints (Joints)
    for i, kpt in enumerate(keypoints):
        y, x, score = kpt
        if score > score_threshold:
            center = (int(x * width), int(y * height))
            
            if i == 0: 
                # Head - Larger and Thicker
                cv2.circle(output_image, center, 30, (255, 255, 255), 3) 
            elif i in [1, 2, 3, 4]: 
                continue 
            else:
                # Joints - Larger (radius 7 instead of 5)
                cv2.circle(output_image, center, 7, (0, 0, 255), -1) 

    return output_image