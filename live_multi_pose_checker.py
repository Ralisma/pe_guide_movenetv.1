# File: live_multi_pose_checker.py

import cv2
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Import from your other files ---
from model import movenet, input_size
from visualization import draw_prediction_on_image

# --- Helper Functions (from previous script) ---
def get_keypoints(image):
    """Runs MoveNet on a single image and returns the keypoints."""
    # Ensure image is a tensor
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        
    # Ensure 3 channels
    if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)
        
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    
    keypoints_with_scores = movenet(input_image)
    return keypoints_with_scores

def flatten_keypoints(keypoints_with_scores):
    """Flattens the keypoints to a 1D vector for comparison."""
    keypoints = np.squeeze(keypoints_with_scores)[:, :2] # Take only (y, x)
    return keypoints.flatten()

# --- 1. BUILD THE REFERENCE POSE LIBRARY ---

def load_reference_poses(image_dir):
    """
    Loads all images from the directory, runs inference, and creates
    an averaged keypoint vector for each pose class.
    """
    print(f"Loading reference poses from: {image_dir}")
    
    # Use logic from your image_loader.py
    try:
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"Directory not found: '{image_dir}'")
        print("Please check that the path is correct.")
        return None

    image_paths = [os.path.join(image_dir, f) for f in image_files]

    # Categorize image paths by class
    classes_files = {
        'ars': [p for p in image_paths if 'ars' in os.path.basename(p)],
        'shf': [p for p in image_paths if 'shf' in os.path.basename(p)],
        'sqs': [p for p in image_paths if 'sqs' in os.path.basename(p)],
        'start': [p for p in image_paths if 'start' in os.path.basename(p)]
    }

    reference_poses = {} # This will store the final {name: vector}
    
    for class_name, paths in classes_files.items():
        if not paths:
            print(f"Warning: No images found for class '{class_name}'.")
            continue
            
        class_vectors = []
        print(f"Processing class: '{class_name}' ({len(paths)} images)")
        
        for image_path in paths:
            try:
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                
                # Get keypoints and flatten
                keypoints = get_keypoints(image)
                vector = flatten_keypoints(keypoints)
                class_vectors.append(vector)
            except Exception as e:
                print(f"  Could not process {image_path}: {e}")
        
        if class_vectors:
            # Average all vectors for this class to get a robust "master" pose
            mean_vector = np.mean(class_vectors, axis=0)
            # Reshape for cosine_similarity (expects 2D)
            reference_poses[class_name] = mean_vector.reshape(1, -1)
            
    print("Reference pose library loaded successfully.")
    print(f"Found poses for: {list(reference_poses.keys())}")
    return reference_poses

# --- 2. MAIN APPLICATION ---

# --- CONFIGURATION ---
# This is the path you provided
IMAGE_DATABASE_DIR = r'D:\182peguide\peguidenew\imagedatabase' 
SIMILARITY_THRESHOLD = 0.85 # Tune this value (0.0 to 1.0)

# --- LOAD POSES ---
reference_poses = load_reference_poses(IMAGE_DATABASE_DIR)
if not reference_poses:
    print("Exiting. Could not load reference poses.")
    exit()

# --- SETUP WEBCAM ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("\nStarting webcam feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (MoveNet)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- Run inference on the live frame ---
    live_keypoints_with_scores = get_keypoints(frame_rgb)
    
    # --- Find the best matching pose ---
    live_vector_2d = flatten_keypoints(live_keypoints_with_scores).reshape(1, -1)
    
    best_match_pose = "None"
    best_match_score = 0.0

    for pose_name, ref_vector_2d in reference_poses.items():
        score = cosine_similarity(ref_vector_2d, live_vector_2d)[0][0]
        
        if score > best_match_score:
            best_match_score = score
            best_match_pose = pose_name

    # --- DRAW VISUALIZATIONS ---
    
    # Draw the skeleton on the frame
    output_overlay = draw_prediction_on_image(
        frame, # Use the original BGR frame for drawing
        live_keypoints_with_scores
    )
    
    # Set display text and color based on match
    if best_match_score > SIMILARITY_THRESHOLD:
        feedback_text = f"MATCH: {best_match_pose.upper()} ({best_match_score:.2f})"
        color = (0, 255, 0) # Green
    else:
        feedback_text = "POSE: NO MATCH"
        color = (0, 0, 255) # Red

    # Put the text on the image
    cv2.putText(
        output_overlay,
        feedback_text,
        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
    )
    
    # Display the resulting frame
    cv2.imshow('Live Pose Checker', output_overlay)

    if cv2.waitKey(1) == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.")