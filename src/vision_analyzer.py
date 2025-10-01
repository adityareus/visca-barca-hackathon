# src/vision_analyzer.py - Handles scene analysis using Cerebras + Meta LLama Vision

from cerebras.cloud.sdk import Cerebras

class VisionAnalyzer:
    def __init__(self, api_key):
        self.client = Cerebras(api_key)
        self.model = "llama-3.3-70b-vision"

        self.navigation_prompt = """You are a vision assistant helping a blind person navigate their surroundings. 
Analyze this image and provide:
1. Immediate hazards or obstacles in the path (walls, furniture, stairs, people, objects)
2. Clear directions about what's ahead, to the left, and to the right
3. Any important objects or landmarks nearby
4. Safe navigation suggestions

Keep your response concise (2-3 sentences), clear, and actionable. Focus on what's most important for safe navigation.
Speak in second person (e.g., "You have a chair 2 feet ahead of you")."""


    # TODO: encode_image -> analyze the scene using the prompt/custom prompt
    