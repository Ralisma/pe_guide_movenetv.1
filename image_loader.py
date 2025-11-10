import tensorflow as tf
import os

# Directory containing the images
image_dir = 'imagedatabase'

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create image paths
image_paths = [os.path.join(image_dir, f) for f in image_files]

# Categorize image paths by class
ars_paths = [path for path in image_paths if 'ars' in os.path.basename(path)]
shf_paths = [path for path in image_paths if 'shf' in os.path.basename(path)]
sqs_paths = [path for path in image_paths if 'sqs' in os.path.basename(path)]
start_paths = [path for path in image_paths if 'start' in os.path.basename(path)]

# Dictionary of classes and their paths
classes = {
    'ars': ars_paths,
    'shf': shf_paths,
    'sqs': sqs_paths,
    'start': start_paths
}

# Function to load an image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    return image

# Example: Load the first image from each class
for class_name, paths in classes.items():
    if paths:
        image_path = paths[0]
        image = load_image(image_path)
        print(f"Loaded image from class '{class_name}': {image_path}")
    else:
        print(f"No images found for class '{class_name}'.")
