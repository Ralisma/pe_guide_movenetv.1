import time

class PoseVerifier:
    def __init__(self, pose_checker):
        self.pose_checker = pose_checker
        self.verification_count = 0
        self.required_verifications = 3

    def verify_pose(self, target_pose):
        self.verification_count = 0
        while self.verification_count < self.required_verifications:
            # Get current pose from pose_checker
            current_pose = self.pose_checker.get_current_pose()
            if current_pose == target_pose:
                self.verification_count += 1
                print(f"Verification {self.verification_count}/{self.required_verifications} successful")
            else:
                print("Pose not matched, try again")
                time.sleep(1)  # Brief pause before next check
        return True

    def reset(self):
        self.verification_count = 0
