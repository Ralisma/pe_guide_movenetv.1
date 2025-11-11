import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2
import urllib.request

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

# --- REDUNDANT MODEL LOADING CODE HAS BEEN REMOVED ---
# (The model is correctly loaded in model.py)


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

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                      height,
                                      width,
                                      keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) where:
      keypoints_xy: A numpy array shape [num_keypoints, 2].
      edges_xy: A list of tuples with keypoint indices.
      edge_colors: A list with the same length as edges_xy with color names.
  """
  keypoints_all = np.squeeze(keypoints_with_scores)
  keypoints_xy = keypoints_all[:, [1, 0]]  # Swap x and y for visualization
  keypoints_scores = keypoints_all[:, 2]

  # Convert keypoints to pixel coordinates
  keypoints_xy = keypoints_xy * np.array([width, height])

  # --- FIX 1: Create a new variable for the filtered scatter plot points ---
  # We keep the original `keypoints_xy` (with 17 points) for edge drawing.
  display_keypoints_mask = keypoints_scores > keypoint_threshold
  keypoints_for_scat = keypoints_xy[display_keypoints_mask, :]

  # Get edges
  edges_xy = []
  edge_colors = []
  for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
    # --- FIX 2: Check scores but use simple indexing on the full array ---
    if (keypoints_scores[edge[0]] > keypoint_threshold and
        keypoints_scores[edge[1]] > keypoint_threshold):
      # Use simple, direct indexing on the full 17-point array
      edges_xy.append((keypoints_xy[edge[0]], keypoints_xy[edge[1]]))
      edge_colors.append(color)

  # --- FIX 3: Return the filtered points for the scatter plot ---
  return keypoints_for_scat, edges_xy, edge_colors

def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on an image.

  Args:
    image: A numpy array with shape [height, width, 3] representing the pixel
      values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, 3] with the keypoints and
    the bounding box drawn on the image.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  # Note: keypoint_locs now refers to the filtered points (keypoints_for_scat)
  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  
  # This 'if' statement is correctly indented
  if keypoint_locs.size > 0:
    scat.set_offsets(keypoint_locs)

  # All the following lines are also correctly indented
  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  # Get the RGBA buffer from the figure
  buf = fig.canvas.buffer_rgba()
  # Convert the buffer to a NumPy array
  image_from_plot_rgba = np.asarray(buf)
  # Convert RGBA to RGB (which the rest of the code expects)
  image_from_plot = cv2.cvtColor(image_from_plot_rgba, cv2.COLOR_RGBA2RGB)
  plt.close(fig)
  if output_image_height is not None:
    output_image_height = int(output_image_height)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(int(output_image_height * aspect_ratio),
                                output_image_height),
         interpolation=cv2.INTER_CUBIC)
  
  # This return statement is now correctly indented
  return image_from_plot