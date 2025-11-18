import pyttsx3
import time
import pygame
import threading

class PoseInstructions:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1)
        pygame.mixer.init()

    def speak(self, text):
        """Speak text in a non-blocking way using a daemon thread."""
        threading.Thread(target=self._speak_blocking, args=(text,), daemon=True).start()

    def _speak_blocking(self, text):
        """Blocking speak function for the thread."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def play_sound(self, sound_file):
        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing sound: {e}. Using text notification instead.")
            print(f"Sound: {sound_file}")

    def guide_pose(self, pose_name, status):
        if status == "start":
            self.speak(f"Get ready for {pose_name} pose.")
            # Sound file removed as per user request
        elif status == "almost":
            self.speak("Almost there, adjust your pose.")
        elif status == "not_quite":
            self.speak("Not quite there. Try to match the pose.")
        elif status == "good":
            self.speak("Good job! Pose completed.")
        elif status == "rest":
            self.speak("Take a 5 second break to rest.")
            # time.sleep(5)  <-- This was already correctly removed

    def run_sequence(self):
        # This is just a simulation and is not used by main.py
        poses = ["start", "ars", "shf", "sqs", "start"]
        for pose in poses:
            self.guide_pose(pose, "start")
            time.sleep(2)
            self.guide_pose(pose, "good")
            if pose != "start":
                self.guide_pose(pose, "rest")
        self.speak("Program completed. Well done!")

if __name__ == "__main__":
    instructor = PoseInstructions()
    print("Running instruction sequence simulation...")
    instructor.run_sequence()
