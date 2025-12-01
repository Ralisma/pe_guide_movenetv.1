# File: main.py
# Description: Optimized Main Program with Underscore Fix and Updated Thresholds

# --- 1. IMPORTS ---
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

from instructions import PoseInstructions
from visualization import draw_prediction_on_image
from model import movenet, input_size

# --- 2. HELPER FUNCTIONS ---

def load_reference_poses(image_database_dir):
    reference_poses = {}
    if not os.path.isdir(image_database_dir):
        print(f"Error: Image database directory not found at: {image_database_dir}")
        return None

    print(f"Loading all reference images from: {image_database_dir}")
    # Iterate over folder names (which act as pose names)
    for pose_name in os.listdir(image_database_dir):
        pose_dir = os.path.join(image_database_dir, pose_name)
        if os.path.isdir(pose_dir):
            image_files = [f for f in os.listdir(pose_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if not image_files: continue

            print(f"  Loaded class: {pose_name} ({len(image_files)} images)")
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(pose_dir, image_file)
                # Create a key. Logic: pose_name + unique_separator + index
                reference_key = f"{pose_name}|{i}" 
                try:
                    image = tf.io.read_file(image_path)
                    image = tf.image.decode_jpeg(image)
                    keypoints = get_keypoints(image)
                    flat_keypoints = flatten_keypoints(keypoints)
                    reference_poses[reference_key] = flat_keypoints.reshape(1, -1)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
            
    return reference_poses

def get_keypoints(image):
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
    if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    keypoints_with_scores = movenet(input_image)
    return keypoints_with_scores

def flatten_keypoints(keypoints_with_scores):
    keypoints = np.squeeze(keypoints_with_scores)[:, :2]
    return keypoints.flatten()

# --- 3. MAIN PROGRAM CLASS ---

class GuidedPoseProgram:
    def __init__(self):
        print("Initializing program variables...")
        self.instructor = PoseInstructions()
        self.reference_poses = None
        self.cap = None
        
        # UPDATED THRESHOLDS: Added generic fallback and specific new poses
        # Lowered strictly to 0.85 for testing, can increase to 0.90 later
        self.similarity_thresholds = {
            "default": 0.85, 
            "start": 0.90,
            "warrior_left": 0.80, # Complex poses might need lower thresholds
            "warrior_right": 0.80,
            "table_left": 0.85,
            "table_right": 0.85
        }

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_database_dir = os.path.join(base_dir, 'imagedatabase')

        # Load Sequence
        self.poses_sequence = []
        sequence_file_path = os.path.join(base_dir, 'sequence.txt')
        try:
            with open(sequence_file_path, 'r') as f:
                for line in f:
                    pose_name = line.strip()
                    if pose_name and not pose_name.startswith('#'):
                        self.poses_sequence.append(pose_name)
        except FileNotFoundError:
            print("Sequence file not found, please create sequence.txt")

        self.current_pose_index = 0
        self.pose_start_time = None
        self.required_hold_time = 3.0
        self.last_feedback_time = 0
        self.feedback_cooldown = 3
        self.rest_start_time = 0
        self.rest_duration = 5
        
        # Distance Variables
        self.optimal_pose_height_range = (0.4, 0.9)  
        self.last_distance_feedback_time = 0
        self.distance_feedback_cooldown = 3.0 
        
        # No User Skip Variables
        self.no_user_start_time = None
        self.no_user_skip_time = 5.0 
        self.no_user_countdown_spoken = {5: False, 4: False, 3: False, 2: False, 1: False}

        self.program_started = False

    def initialize(self):
        self.reference_poses = load_reference_poses(self.image_database_dir)
        if not self.reference_poses: return False

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened(): return False
        self.program_started = True
        return True

    def get_best_match(self, live_keypoints):
        live_vector = flatten_keypoints(live_keypoints).reshape(1, -1)
        best_match_pose = "None"
        best_match_score = 0.0

        for reference_key, ref_vector in self.reference_poses.items():
            score = cosine_similarity(ref_vector, live_vector)[0][0]
            if score > best_match_score:
                best_match_score = score
                # FIX: Using '|' as separator to safely extract pose name
                # even if the pose name contains underscores
                best_match_pose = reference_key.split('|')[0]
                
        return best_match_pose, best_match_score

    def give_corrective_feedback(self, target_pose, current_pose, score):
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            # Get specific threshold or default
            threshold = self.similarity_thresholds.get(target_pose, self.similarity_thresholds["default"])
            
            if current_pose == target_pose and score > (threshold - 0.1):
                self.instructor.guide_pose(target_pose, "almost")
            else:
                self.instructor.guide_pose(target_pose, "not_quite")
            self.last_feedback_time = current_time

    def get_pose_size(self, keypoints_with_scores, min_confidence=0.2):
        keypoints = np.squeeze(keypoints_with_scores)
        y_coords = keypoints[:, 0]
        scores = keypoints[:, 2]
        
        visible_y_coords = y_coords[scores > min_confidence]
        if visible_y_coords.size < 2: return 0.0, False 

        min_y = np.min(visible_y_coords)
        max_y = np.max(visible_y_coords)
        pose_height = max_y - min_y
        
        all_keypoints_visible = np.sum(scores > min_confidence) > 5
        return pose_height, all_keypoints_visible

    def give_distance_feedback(self, pose_height, all_keypoints_visible):
        current_time = time.time()
        if current_time - self.last_distance_feedback_time < self.distance_feedback_cooldown:
            return  

        min_height, max_height = self.optimal_pose_height_range
        if pose_height == 0.0: return 

        if pose_height < min_height:
            self.instructor.speak("Too far. Move closer.")
            self.last_distance_feedback_time = current_time
        elif pose_height > max_height:
            self.instructor.speak("Too close. Move back.")
            self.last_distance_feedback_time = current_time

    def run_program(self):
        if not self.program_started: return
        print("Starting program. Press 'q' to quit.")
        self.instructor.speak("Welcome. Get in the start pose.")

        while True:
            # --- CHECK IF PROGRAM FINISHED ---
            if self.current_pose_index >= len(self.poses_sequence):
                break
            
            target_pose = self.poses_sequence[self.current_pose_index]

            # --- REST LOGIC ---
            if self.rest_start_time > 0:
                rest_elapsed = time.time() - self.rest_start_time
                if rest_elapsed < self.rest_duration:
                    ret, frame = self.cap.read()
                    if not ret: break
                    cv2.putText(frame, f"Rest: {int(self.rest_duration - rest_elapsed)}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Guided Pose Program', frame)
                    if cv2.waitKey(1) == ord('q'): break
                    continue
                else:
                    self.rest_start_time = 0
                    if self.current_pose_index < len(self.poses_sequence):
                        self.instructor.guide_pose(self.poses_sequence[self.current_pose_index], "start")
                    self.last_feedback_time = time.time()
                    self.last_distance_feedback_time = time.time()

            # --- MAIN LOOP ---
            ret, frame = self.cap.read()
            if not ret: break

            # Process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_keypoints = get_keypoints(frame_rgb)
            
            # Draw Visualization
            output_overlay = draw_prediction_on_image(frame, live_keypoints)

            # --- Distance & No User Check ---
            pose_height, visible = self.get_pose_size(live_keypoints)
            
            # Visual Debug
            dist_color = (0, 0, 255) # Red
            if 0.4 <= pose_height <= 0.9: dist_color = (0, 255, 0) # Green
            cv2.putText(output_overlay, f"Dist: {pose_height:.2f}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)

            self.give_distance_feedback(pose_height, visible)

            # --- No User Logic ---
            if pose_height == 0.0:
                if self.no_user_start_time is None:
                    self.no_user_start_time = time.time()
                    self.instructor.speak("No user. Skipping in 5.")
                    self.no_user_countdown_spoken = {5: True, 4: False, 3: False, 2: False, 1: False}
                else:
                    elapsed_no_user = time.time() - self.no_user_start_time
                    rem = self.no_user_skip_time - elapsed_no_user
                    
                    # Vocal Countdown
                    for i in range(4, 0, -1):
                        if rem <= i and not self.no_user_countdown_spoken[i]:
                            self.instructor.speak(str(i))
                            self.no_user_countdown_spoken[i] = True
                            break
                    
                    cv2.putText(output_overlay, f"Skipping: {int(rem)}s", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    if rem <= 0:
                        self.instructor.speak(f"Skipping {target_pose}.")
                        self.current_pose_index += 1
                        self.pose_start_time = None
                        self.no_user_start_time = None
                        
                        if self.current_pose_index < len(self.poses_sequence):
                            self.instructor.guide_pose(self.poses_sequence[self.current_pose_index], "start")
                        continue
            else:
                if self.no_user_start_time is not None:
                    self.instructor.speak(f"Welcome back. Do {target_pose}")
                    self.no_user_start_time = None

            # --- Pose Matching ---
            if pose_height > 0.0:
                current_pose, score = self.get_best_match(live_keypoints)
                # Look up threshold or use default
                threshold = self.similarity_thresholds.get(target_pose, self.similarity_thresholds["default"])

                if current_pose == target_pose and score > threshold:
                    if self.pose_start_time is None:
                        self.pose_start_time = time.time()
                    else:
                        elapsed = time.time() - self.pose_start_time
                        if elapsed >= self.required_hold_time:
                            self.instructor.guide_pose(target_pose, "good")
                            self.current_pose_index += 1
                            self.pose_start_time = None
                            
                            if self.current_pose_index == len(self.poses_sequence):
                                print("Workout Complete. Speaking ending message...")
                                self.instructor.speak("Thank you for performing. Ending Session Now.")
                                end_timer = time.time()
                                while time.time() - end_timer < 6.0:
                                    ret, frame = self.cap.read()
                                    if ret:
                                        cv2.putText(frame, "Session Ending...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                                        cv2.imshow('Guided Pose Program', frame)
                                        cv2.waitKey(1)
                                break 
                            elif self.current_pose_index < len(self.poses_sequence):
                                self.instructor.guide_pose(self.poses_sequence[self.current_pose_index], "rest")
                                self.rest_start_time = time.time()

                else:
                    if self.pose_start_time is not None:
                        self.pose_start_time = None
                    self.give_corrective_feedback(target_pose, current_pose, score)

                # Status Text
                hold_t = (time.time() - self.pose_start_time) if self.pose_start_time else 0.0
                st_txt = f"{target_pose.upper()} | {current_pose.upper()} ({score:.2f}) | {hold_t:.1f}s"
                col = (0, 255, 0) if (current_pose == target_pose and score > threshold) else (0, 0, 255)
                cv2.putText(output_overlay, st_txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            cv2.imshow('Guided Pose Program', output_overlay)
            if cv2.waitKey(1) == ord('q'): break

        self.cleanup()

    def cleanup(self):
        self.instructor.engine.stop()
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    program = GuidedPoseProgram()
    if program.initialize():
        program.run_program()