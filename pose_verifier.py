import time

class PoseVerifier:
    def __init__(self, pose_checker):
        self.pose_checker = pose_checker
        self.required_hold_time = 3.0  # seconds

    def verify_pose(self, target_pose):
        start_time = None
        while True:
            # Get current pose from pose_checker
            current_pose = self.pose_checker.get_current_pose()
            if current_pose == target_pose:
                if start_time is None:
                    start_time = time.time()
                    print("Pose matched, start holding...")
                else:
                    elapsed = time.time() - start_time
                    print(f"Holding pose: {elapsed:.1f}/{self.required_hold_time} seconds")
                    if elapsed >= self.required_hold_time:
                        print("Pose held successfully!")
                        return True
            else:
                if start_time is not None:
                    print("Pose lost, try again")
                    start_time = None
                time.sleep(0.5)  # Brief pause before next check

    def reset(self):
        pass  # No state to reset
