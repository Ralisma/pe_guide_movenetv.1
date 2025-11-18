# File: main.py
# Description: This is the main file for the Guided Pose Program.
# It runs the webcam, loads reference poses, and guides the user
# through a workout sequence defined in 'sequence.txt'.
#
# NEW: Includes an optimal distance reminder.
# NEW: Includes a 5-second "no user" skip timer with vocal countdown.
# NEW: Includes a "Welcome back" message.

# --- 1. IMPORTS ---
import cv2  # For webcam and image processing
import time  # For timing (hold times, rest, cooldowns)
import os  # For file and path operations (e.g., loading images, sequence file)
import numpy as np  # For numerical operations
import tensorflow as tf  # For running the MoveNet model
from sklearn.metrics.pairwise import cosine_similarity  # For comparing poses

# --- Corrected Imports from our other Python files ---
from instructions import PoseInstructions  # Handles text-to-speech feedback
from visualization import draw_prediction_on_image  # Draws the skeleton on the image
from model import movenet, input_size  # The MoveNet model and its input size

# --- 2. HELPER FUNCTIONS ---
# These functions support the main program, handling pose loading and processing.

def load_reference_poses(image_database_dir):
    """
    Loads ALL reference poses from the image database directory.
    
    This function scans the 'image_database_dir'. Each sub-folder
    (e.g., 'shf', 'sqs') is treated as a pose. Each image inside that
    sub-folder is loaded as an individual reference example.
    
    It creates unique keys for each image, e.g., 'shf_0', 'shf_1'.
    This allows the program to match the user against multiple
    examples of the same pose.
    """
    reference_poses = {}  # Will store "pose_name_0", "pose_name_1", etc.

    if not os.path.isdir(image_database_dir):
        print(f"Error: Image database directory not found at: {image_database_dir}")
        return None

    print(f"Loading all reference images from: {image_database_dir}")
    # Loop through each pose folder (e.g., "start", "ars", "shf")
    for pose_name in os.listdir(image_database_dir):
        pose_dir = os.path.join(image_database_dir, pose_name)
        if os.path.isdir(pose_dir):
            # Find all .jpg or .png images in the pose folder
            image_files = [f for f in os.listdir(pose_dir) if f.endswith(('.jpg', '.png'))]
            if not image_files:
                print(f"Warning: No images found for pose: {pose_name}")
                continue

            print(f"Processing {len(image_files)} images for pose: {pose_name}")

            # Loop through each image file and process it
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(pose_dir, image_file)
                # Create a unique key like "shf_0", "shf_1"
                reference_key = f"{pose_name}_{i}"
                try:
                    # Load and process the image
                    image = tf.io.read_file(image_path)
                    image = tf.image.decode_jpeg(image)
                    keypoints = get_keypoints(image)
                    flat_keypoints = flatten_keypoints(keypoints)

                    # Store the flattened keypoints in the dictionary
                    reference_poses[reference_key] = flat_keypoints.reshape(1, -1)
                    print(f"Loaded reference pose: {reference_key} from {image_file}")
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
            
            if not any(k.startswith(pose_name) for k in reference_poses):
                print(f"Could not load any valid keypoints for {pose_name}")
                
    return reference_poses


def get_keypoints(image):
    """Runs MoveNet on a single image and returns the keypoints."""
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
    if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)

    # Resize and pad the image to the model's expected input size
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run the MoveNet model
    keypoints_with_scores = movenet(input_image)
    return keypoints_with_scores


def flatten_keypoints(keypoints_with_scores):
    """Flattens the keypoints to a 1D vector (34 values) for comparison."""
    # Takes only the (y, x) coordinates and ignores the scores
    keypoints = np.squeeze(keypoints_with_scores)[:, :2]
    return keypoints.flatten()


# --- 3. MAIN PROGRAM CLASS ---
# This class contains the entire logic for the guided pose application.

class GuidedPoseProgram:
    def __init__(self):
        """Initializes the program's state and variables."""
        print("Initializing program variables...")
        self.instructor = PoseInstructions()  # Our text-to-speech helper
        self.reference_poses = None  # Will hold all loaded pose keypoints
        self.cap = None  # Will hold the OpenCV webcam capture object

        # --- Pose-specific similarity thresholds ---
        self.similarity_thresholds = {
            "start": 0.95,
            "ars": 0.95,
            "shf": 0.85,  # Less strict for "shf" pose
            "sqs": 0.95
        }

        # --- Get the directory this script is running in ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_database_dir = os.path.join(base_dir, 'imagedatabase')

        # --- Load workout sequence from 'sequence.txt' ---
        self.poses_sequence = []  # The list of poses in the workout
        sequence_file_path = os.path.join(base_dir, 'sequence.txt')

        try:
            with open(sequence_file_path, 'r') as f:
                for line in f:
                    pose_name = line.strip()  # Remove whitespace/newlines
                    if pose_name and not pose_name.startswith('#'):
                        self.poses_sequence.append(pose_name)
            
            if not self.poses_sequence:
                print("Error: 'sequence.txt' is empty. Please add pose names to it.")
                self.program_started = False
            else:
                print(f"Loaded sequence from {sequence_file_path}: {self.poses_sequence}")

        except FileNotFoundError:
            print(f"Error: 'sequence.txt' not found at {sequence_file_path}")
            print("Please create 'sequence.txt' in the same directory as main.py.")
            self.program_started = False
        # --- End of sequence loading ---

        # --- Program State Variables ---
        self.current_pose_index = 0
        self.pose_start_time = None
        self.required_hold_time = 3.0  # Seconds
        self.last_feedback_time = 0
        self.feedback_cooldown = 3  # Seconds
        self.rest_start_time = 0
        self.rest_duration = 5  # Seconds
        
        # --- Distance Feedback Variables ---
        self.optimal_pose_height_range = (0.4, 0.9)  # Min/Max pose height
        self.last_distance_feedback_time = 0
        self.distance_feedback_cooldown = 10.0  # Seconds
        
        # --- No User Skip Timer ---
        self.no_user_start_time = None
        self.no_user_skip_time = 5.0 # Seconds to wait before skipping
        # NEW: Track spoken countdown numbers
        self.no_user_countdown_spoken = {5: False, 4: False, 3: False, 2: False, 1: False}

        self.program_started = False


    def initialize(self):
        """
        Loads all reference poses and starts the webcam.
        Returns True on success, False on failure.
        """
        print("Initializing Guided Pose Program...")
        self.reference_poses = load_reference_poses(self.image_database_dir)
        if not self.reference_poses:
            print("Failed to load reference poses. Exiting.")
            return False

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return False

        self.program_started = True
        return True


    def get_best_match(self, live_keypoints):
        """
        Compares the user's live keypoints against ALL reference images.
        
        Returns the "base pose name" (e.g., "shf") and the highest
        similarity score found.
        """
        live_vector = flatten_keypoints(live_keypoints).reshape(1, -1)
        best_match_pose = "None"
        best_match_score = 0.0

        for reference_key, ref_vector in self.reference_poses.items():
            score = cosine_similarity(ref_vector, live_vector)[0][0]
            
            if score > best_match_score:
                best_match_score = score
                best_match_pose = reference_key.split('_')[0]
        
        return best_match_pose, best_match_score


    def give_corrective_feedback(self, target_pose, current_pose, score):
        """
        Gives 'almost' or 'not_quite' feedback during a pose attempt.
        Uses a cooldown to avoid spamming the user with audio.
        """
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            threshold = self.similarity_thresholds.get(target_pose, 0.95)
            
            if current_pose == target_pose and score > (threshold - 0.1):
                self.instructor.guide_pose(target_pose, "almost")
            else:
                self.instructor.guide_pose(target_pose, "not_quite")
            
            self.last_feedback_time = current_time


    def get_pose_size(self, keypoints_with_scores, min_confidence=0.1):
        """
        Calculates the apparent height of the pose and checks keypoint visibility.
        Height is returned as a fraction of the frame (0.0 to 1.0).
        """
        keypoints = np.squeeze(keypoints_with_scores)  # Shape (17, 3)
        
        y_coords = keypoints[:, 0]
        scores = keypoints[:, 2]
        
        visible_y_coords = y_coords[scores > min_confidence]
        
        if visible_y_coords.size == 0:
            # No keypoints detected
            return 0.0, False 

        min_y = np.min(visible_y_coords)
        max_y = np.max(visible_y_coords)
        
        pose_height = max_y - min_y
        
        all_keypoints_visible = np.sum(scores > min_confidence) > 10 

        return pose_height, all_keypoints_visible

    def give_distance_feedback(self, pose_height, all_keypoints_visible):
        """Gives feedback if the user is too close or too far."""
        current_time = time.time()
        if current_time - self.last_distance_feedback_time < self.distance_feedback_cooldown:
            return  

        min_height, max_height = self.optimal_pose_height_range

        if pose_height == 0.0:
            return # Don't give feedback if no one is on screen

        if pose_height < min_height:
            self.instructor.speak("You are too far away. Please move closer.")
            self.last_distance_feedback_time = current_time
        
        elif pose_height > max_height or not all_keypoints_visible:
            self.instructor.speak("You are too close. Please move farther back.")
            self.last_distance_feedback_time = current_time

    def run_program(self):
        """
        This is the main loop of the program.
        """
        if not self.program_started:
            print("Initialization failed. Cannot run program.")
            return

        print("Starting program. Press 'q' to quit.")
        self.instructor.speak("Welcome! Let's begin. Get in the start pose.")

        while True:
            # --- Check if Program is Finished ---
            if self.current_pose_index >= len(self.poses_sequence):
                print("Program finished.")
                break

            target_pose = self.poses_sequence[self.current_pose_index]

            # --- A. Handle Rest Period ---
            if self.rest_start_time > 0:
                rest_elapsed = time.time() - self.rest_start_time
                if rest_elapsed < self.rest_duration:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    rest_text = f"Rest: {int(self.rest_duration - rest_elapsed)}s"
                    cv2.putText(frame, rest_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Guided Pose Program', frame)

                    if cv2.waitKey(1) == ord('q'):
                        break
                    continue
                else:
                    self.rest_start_time = 0
                    if self.current_pose_index < len(self.poses_sequence):
                        next_pose = self.poses_sequence[self.current_pose_index]
                        self.instructor.guide_pose(next_pose, "start")
                    self.last_feedback_time = time.time()
                    self.last_distance_feedback_time = time.time()

            # --- B. Main Pose Logic ---
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_keypoints = get_keypoints(frame_rgb)
            
            # Draw the skeleton first, so text can go on top
            output_overlay = draw_prediction_on_image(frame, live_keypoints)

            # --- C. Distance & No-User Checks ---
            pose_height, all_keypoints_visible = self.get_pose_size(live_keypoints)
            self.give_distance_feedback(pose_height, all_keypoints_visible)
            
            # --- NEW: No User Detected Skip Logic ---
            if pose_height == 0.0:
                if self.no_user_start_time is None:
                    # Start the timer and speak the first number
                    print("No user detected, starting skip timer...")
                    self.no_user_start_time = time.time()
                    self.instructor.speak("No user detected. Skipping in 5")
                    self.no_user_countdown_spoken = {5: True, 4: False, 3: False, 2: False, 1: False}
                else:
                    # Timer is running
                    elapsed_no_user = time.time() - self.no_user_start_time
                    time_remaining = self.no_user_skip_time - elapsed_no_user

                    # --- NEW: Vocal Countdown Logic ---
                    if time_remaining <= 4 and not self.no_user_countdown_spoken[4]:
                        self.instructor.speak("4")
                        self.no_user_countdown_spoken[4] = True
                    elif time_remaining <= 3 and not self.no_user_countdown_spoken[3]:
                        self.instructor.speak("3")
                        self.no_user_countdown_spoken[3] = True
                    elif time_remaining <= 2 and not self.no_user_countdown_spoken[2]:
                        self.instructor.speak("2")
                        self.no_user_countdown_spoken[2] = True
                    elif time_remaining <= 1 and not self.no_user_countdown_spoken[1]:
                        self.instructor.speak("1")
                        self.no_user_countdown_spoken[1] = True
                    # --- End of Vocal Countdown ---
                    
                    # Display text countdown on screen
                    countdown_text = f"No user. Skipping in: {int(time_remaining)}s"
                    cv2.putText(output_overlay, countdown_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange
                    
                    if time_remaining <= 0:
                        # Time is up, skip the pose
                        print(f"No user detected for {self.no_user_skip_time}s. Skipping pose.")
                        self.instructor.speak(f"Skipping {target_pose}.")
                        
                        # --- Skip to next pose
                        self.current_pose_index += 1
                        self.pose_start_time = None  # Reset pose hold timer
                        self.no_user_start_time = None # Reset skip timer
                        self.no_user_countdown_spoken = {5: False, 4: False, 3: False, 2: False, 1: False} # Reset flags
                        
                        if self.current_pose_index < len(self.poses_sequence):
                            # Announce the new pose
                            next_pose = self.poses_sequence[self.current_pose_index]
                            self.instructor.guide_pose(next_pose, "start")
                        
                        continue # Skip the rest of this loop iteration

            else:
                # User is detected
                if self.no_user_start_time is not None:
                    # --- NEW: Welcome Back Message ---
                    # This means the timer was active, and the user returned
                    print("User re-detected, stopping skip timer.")
                    self.instructor.speak(f"Welcome back. Do {target_pose}")
                    self.no_user_start_time = None
                    self.no_user_countdown_spoken = {5: False, 4: False, 3: False, 2: False, 1: False} # Reset flags
            # --- End of No User Logic ---


            # --- D. Pose Matching Logic ---
            # (Only run if a user is present)
            if pose_height > 0.0:
                current_pose, score = self.get_best_match(live_keypoints)

                # Get the correct similarity threshold for this pose
                threshold = self.similarity_thresholds.get(target_pose, 0.95)

                # 1. If POSE IS A MATCH
                if current_pose == target_pose and score > threshold:
                    if self.pose_start_time is None:
                        self.pose_start_time = time.time()
                        print("Pose matched, start holding...")
                    else:
                        elapsed = time.time() - self.pose_start_time
                        print(f"Holding pose: {elapsed:.1f}/{self.required_hold_time} seconds")
                        
                        if elapsed >= self.required_hold_time:
                            # POSE COMPLETED!
                            self.instructor.guide_pose(target_pose, "good")
                            self.current_pose_index += 1
                            self.pose_start_time = None

                            if self.current_pose_index == len(self.poses_sequence):
                                self.instructor.speak("Program completed. Well done!")
                            elif self.current_pose_index < len(self.poses_sequence):
                                self.instructor.guide_pose(target_pose, "rest")
                                self.rest_start_time = time.time()

                # 2. If POSE IS NOT A MATCH
                else:
                    if self.pose_start_time is not None:
                        print("Pose lost, try again")
                        self.pose_start_time = None
                    
                    self.give_corrective_feedback(target_pose, current_pose, score)

                # --- E. Display Status Info On Screen ---
                hold_time = (time.time() - self.pose_start_time) if self.pose_start_time else 0.0
                status_text = f"Target: {target_pose.upper()} | Detected: {current_pose.upper()} ({score:.2f}) | Held: {hold_time:.1f}/{self.required_hold_time}s"
                
                color = (0, 255, 0) if (current_pose == target_pose and score > threshold) else (0, 0, 255)
                
                cv2.putText(output_overlay, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Show the final image
            cv2.imshow('Guided Pose Program', output_overlay)

            # Check for 'q' key to quit
            if cv2.waitKey(1) == ord('q'):
                break

        # --- End of main loop ---
        self.cleanup()


    def cleanup(self):
        """Releases the webcam and closes all OpenCV windows."""
        self.instructor.engine.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Program cleaned up.")


# --- 4. RUN THE PROGRAM ---
if __name__ == "__main__":
    program = GuidedPoseProgram()
    
    if program.initialize():
        program.run_program()