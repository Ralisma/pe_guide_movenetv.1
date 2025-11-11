import cv2
import time
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# --- Corrected Imports ---
from instructions import PoseInstructions
from visualization import draw_prediction_on_image
from model import movenet, input_size

# --- Helper Functions (Moved from live_multi_pose_checker.py) ---

def load_reference_poses(image_database_dir):
    """Loads reference poses from the image database."""
    reference_poses = {}
    
    if not os.path.isdir(image_database_dir):
        print(f"Error: Image database directory not found at: {image_database_dir}")
        return None

    print(f"Loading reference images from: {image_database_dir}")
    for pose_name in os.listdir(image_database_dir):
        pose_dir = os.path.join(image_database_dir, pose_name)
        if os.path.isdir(pose_dir):
            image_files = [f for f in os.listdir(pose_dir) if f.endswith(('.jpg', '.png'))]
            if not image_files:
                print(f"Warning: No images found for pose: {pose_name}")
                continue

            image_path = os.path.join(pose_dir, image_files[0])
            try:
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image)
                keypoints = get_keypoints(image)
                reference_poses[pose_name] = flatten_keypoints(keypoints).reshape(1, -1)
                print(f"Loaded reference pose: {pose_name}")
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                
    return reference_poses

def get_keypoints(image):
    """Runs MoveNet on a single image and returns the keypoints."""
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
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

# --- Main Program Class ---

class GuidedPoseProgram:
    def __init__(self):
        self.instructor = PoseInstructions()
        self.reference_poses = None
        self.cap = None
        self.similarity_threshold = 0.85
        
        # --- FIX: Use a relative path ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_database_dir = os.path.join(base_dir, 'imagedatabase')
        
        self.poses_sequence = ["start", "ars", "shf", "sqs", "start"]
        self.current_pose_index = 0
        self.verification_count = 0
        self.required_verifications = 3
        self.last_feedback_time = 0
        self.feedback_cooldown = 3  # seconds
        self.rest_start_time = 0
        self.rest_duration = 5  # seconds
        self.program_started = False

    def initialize(self):
        print("Initializing Guided Pose Program...")
        self.reference_poses = load_reference_poses(self.image_database_dir)
        if not self.reference_poses:
            print("Failed to load reference poses. Exiting.")
            return False
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return False
        
        self.program_started = True
        return True

    def get_best_match(self, live_keypoints):
        """Finds the best matching pose from the reference library."""
        live_vector = flatten_keypoints(live_keypoints).reshape(1, -1)
        best_match_pose = "None"
        best_match_score = 0.0

        for pose_name, ref_vector in self.reference_poses.items():
            score = cosine_similarity(ref_vector, live_vector)[0][0]
            if score > best_match_score:
                best_match_score = score
                best_match_pose = pose_name
        
        return best_match_pose, best_match_score

    def give_corrective_feedback(self, target_pose, current_pose, score):
        """Gives 'almost' or 'not_quite' feedback during a pose attempt."""
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            if current_pose == target_pose and score > (self.similarity_threshold - 0.1):
                self.instructor.guide_pose(target_pose, "almost")
            else:
                self.instructor.guide_pose(target_pose, "not_quite")
            self.last_feedback_time = current_time

    def run_program(self):
        if not self.program_started:
            print("Initialization failed. Cannot run program.")
            return

        print("Starting program. Press 'q' to quit.")
        self.instructor.speak("Welcome! Let's begin. Get in the start pose.")

        while True:
            if self.current_pose_index >= len(self.poses_sequence):
                print("Program finished.")
                break

            target_pose = self.poses_sequence[self.current_pose_index]

            # --- Handle Rest Period (Non-blocking) ---
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
                    # After rest, announce the next pose
                    if self.current_pose_index < len(self.poses_sequence):
                        next_pose = self.poses_sequence[self.current_pose_index]
                        self.instructor.guide_pose(next_pose, "start")
                    self.last_feedback_time = time.time() # Reset feedback timer

            # --- Main Pose Logic ---
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_keypoints = get_keypoints(frame_rgb)
            
            current_pose, score = self.get_best_match(live_keypoints)
            
            output_overlay = draw_prediction_on_image(frame, live_keypoints)

            # --- Verification & Feedback Logic ---
            if current_pose == target_pose and score > self.similarity_threshold:
                self.verification_count += 1
                
                if self.verification_count >= self.required_verifications:
                    self.instructor.guide_pose(target_pose, "good")
                    self.current_pose_index += 1
                    self.verification_count = 0
                    
                    if self.current_pose_index == len(self.poses_sequence):
                        self.instructor.speak("Program completed. Well done!")
                    elif self.current_pose_index < len(self.poses_sequence):
                        # Start rest *before* announcing the next pose
                        self.instructor.guide_pose(target_pose, "rest")
                        self.rest_start_time = time.time()
            else:
                self.verification_count = 0
                self.give_corrective_feedback(target_pose, current_pose, score)

            # Display info
            status_text = f"Target: {target_pose.upper()} | Detected: {current_pose.upper()} ({score:.2f}) | Held: {self.verification_count}/{self.required_verifications}"
            color = (0, 255, 0) if (current_pose == target_pose and score > self.similarity_threshold) else (0, 0, 255)
            cv2.putText(output_overlay, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Guided Pose Program', output_overlay)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Program cleaned up.")

if __name__ == "__main__":
    program = GuidedPoseProgram()
    if program.initialize():
        program.run_program()