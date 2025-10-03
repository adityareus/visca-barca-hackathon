# src/main.py - application start

import os
import sys
from dotenv import load_dotenv
from vision_analyzer import VisionAnalyzer
import cv2
import time


# Load the ENV variables
load_dotenv()

class LLamaDaredevil():
    def __init__(self):
        self.api_key = os.getenv("CEREBRAS_API_KEY")

        if not self.api_key:
            print("Error: CEREBRAS_API_KEY not found in env variables")
            print("Please set it in .env file or export it in your terminal")
            sys.exit(1)

        os.environ["CEREBRAS_API_KEY"] = self.api_key
        self.vision_analyzer = VisionAnalyzer()
        self.analysis_interval = 10 #secs
        self.last_analysis_time = 0
        self.last_analysis = ""

    
    def run(self):
        # Application loop
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            print("Make sure your webcam is connected and not in use in other application")
            sys.exit(1)
        
        print("=" * 60)
        print("Llama Daredevil - Vision Assistant MVP")
        print("=" * 60)
        print("\nControls:")
        print("  Q - Quit application")
        print("  S - Repeat last analysis")
        print("\nAnalyzing scene every 3 seconds...")
        print("=" * 60)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                cv2.imshow('Llama Daredevil - Press Q to quit', frame)

                # Analyze scene at time intervals
                current_time = time.time()
                if current_time - self.last_analysis_time >= self.analysis_interval:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Analyzing scene...")

                    _analysis = self.vision_analyzer.analyze_scene(frame)

                    if _analysis and _analysis != self.last_analysis:
                        print(f"Scene Analysis: {_analysis}")
                        # Add speech
                        self.last_analysis = _analysis
                    
                    self.last_analysis_time = current_time
                
                # Key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nShutting down...")
                    break
                elif key == ord('s'):
                    if self.last_description:
                        print(f"Repeating: {self.last_description}")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Application closed")


if __name__ == "__main__":
    app = LLamaDaredevil()
    app.run()