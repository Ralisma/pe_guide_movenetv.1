import pyttsx3
import time
from playsound import playsound

class PoseInstructions:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def play_sound(self, sound_file):
        try:
            playsound(sound_file)
        except Exception as e:
            print(f"Error playing sound: {e}. Using text notification instead.")
            print(f"Sound: {sound_file}")

    def guide_pose(self, pose_name, status):
        if status == "start":
            self.speak(f"Get ready for {pose_name} pose.")
            self.play_sound("start_sound.wav")  # Assume you have a sound file
        elif status == "almost":
            self.speak("Almost there, adjust your pose.")
        elif status == "not_quite":
            self.speak("Not quite there yet, try again.")
        elif status == "good":
            self.speak("Good job! Pose completed.")
        elif status == "rest":
            self.speak("Take a 5 second break to rest.")
            time.sleep(5)

    def run_sequence(self):
        poses = ["start", "ars", "shf", "sqs", "start"]
        for pose in poses:
            self.guide_pose(pose, "start")
            # Here you would integrate with pose checking logic
            # For now, simulate pose verification
            verified = False
            attempts = 0
            while not verified and attempts < 3:
                # Simulate pose checking
                # In real implementation, this would check the pose 3 times
                self.guide_pose(pose, "almost")  # Example feedback
                time.sleep(2)  # Simulate time for pose adjustment
                verified = True  # Assume verified for demo
                attempts += 1
            if verified:
                self.guide_pose(pose, "good")
            else:
                self.guide_pose(pose, "not_quite")
            if pose != "start":  # No rest after final start
                self.guide_pose(pose, "rest")
        self.speak("Program completed. Well done!")

if __name__ == "__main__":
    instructor = PoseInstructions()
    instructor.run_sequence()
