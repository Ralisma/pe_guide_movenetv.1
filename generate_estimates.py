# File: generate_estimates.py
import cv2
import os
import tensorflow as tf
import numpy as np

# Import your existing model and visualization tools
from model import movenet, input_size
from visualization import draw_prediction_on_image

def get_keypoints_for_debug(image):
    """
    Helper to prepare image for model and get keypoints.
    """
    # Convert numpy array to tensor
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
    
    # Check if we need to expand dims (1, H, W, C)
    input_image = tf.expand_dims(image, axis=0)
    
    # Resize to what the model expects (192x192)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    
    # Run Inference
    keypoints_with_scores = movenet(input_image)
    return keypoints_with_scores

def process_database():
    # 1. Define Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'imagedatabase')
    output_dir = os.path.join(base_dir, 'model_estimates')

    # 2. Check if source exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found at {source_dir}")
        return

    # 3. Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Processing images from '{source_dir}' to '{output_dir}'...")
    print("-" * 50)

    # 4. Loop through folders (ars, shf, sqs, start)
    for pose_name in os.listdir(source_dir):
        pose_dir = os.path.join(source_dir, pose_name)
        
        # Skip if not a folder
        if not os.path.isdir(pose_dir):
            continue

        # Create corresponding folder in output
        output_pose_dir = os.path.join(output_dir, pose_name)
        os.makedirs(output_pose_dir, exist_ok=True)

        # Loop through images in the folder
        image_files = [f for f in os.listdir(pose_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for file_name in image_files:
            file_path = os.path.join(pose_dir, file_name)
            
            # Read image using OpenCV
            frame = cv2.imread(file_path)
            if frame is None:
                print(f"Skipping corrupt file: {file_name}")
                continue

            # Convert BGR (OpenCV standard) to RGB (Model standard)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                # Run Model
                keypoints = get_keypoints_for_debug(frame_rgb)

                # Draw Skeleton on the ORIGINAL BGR image
                # (We use the original frame so colors look correct when saved)
                annotated_image = draw_prediction_on_image(frame, keypoints)

                # Save to output folder
                save_path = os.path.join(output_pose_dir, f"ESTIMATE_{file_name}")
                cv2.imwrite(save_path, annotated_image)
                print(f"Processed: {pose_name}/{file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print("-" * 50)
    print(f"Done! Check the '{output_dir}' folder to see how the AI reads your photos.")

if __name__ == "__main__":
    process_database()