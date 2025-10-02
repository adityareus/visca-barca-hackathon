# src/vision_analyzer.py - Handles scene analysis using Cerebras + Meta LLama Vision
from cerebras.cloud.sdk import Cerebras
import cv2
from PIL import Image
import io
import base64

# TODO: Currently Cerebras doesn't allow vision analysis model.. Have to use local Vision Model in front of it
class VisionAnalyzer:
    def __init__(self):
        self.client = Cerebras()
        self.model = "llama-3.3-70b"

        self.navigation_prompt = """You are a vision assistant helping a blind person navigate their surroundings. 
            Analyze this image and provide:
            1. Immediate hazards or obstacles in the path (walls, furniture, stairs, people, objects)
            2. Clear directions about what's ahead, to the left, and to the right
            3. Any important objects or landmarks nearby
            4. Safe navigation suggestions

            Keep your response concise (2-3 sentences), clear, and actionable. Focus on what's most important for safe navigation.
            Speak in second person (e.g., "You have a chair 2 feet ahead of you")."""


    def encode_image(self, frame):
        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        pil_img.thumbnail((800, 800), Image.Resampling.LANCZOS)

        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    
    def analyze_scene(self, frame):
        try:
            
            # Encode img
            encoded_img = self.encode_image(frame)

            # Call Cerebras API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.navigation_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{encoded_img}"
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=150
            )

            # Get the analysis
            analysis = response.choices[0].message.content.strip()
            return analysis

        except Exception as e:
            error_msg = f"Error analyzing scene: {str(e)}"
            print(error_msg)
            return None

