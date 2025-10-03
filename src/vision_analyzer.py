# src/vision_analyzer.py - Handles scene analysis using Cerebras + Meta LLama Vision
from cerebras.cloud.sdk import Cerebras
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
import cv2
from PIL import Image
import io
import base64

# NOTE: Need to use your own Hugging face token to use Llama-Vision model. 
# Use > `huggingface-cli login` in terminal and add your HF Token

class VisionAnalyzer:
    def __init__(self):
        print("Powering Llama-Daredevil with Llama models.....")
        self.client = Cerebras()
        self.cerebras_model = "llama-3.3-70b"

        print("Loading Llama 3.2 Vision model (this may take a few minutes first time)...")
        self.vlm = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        try:
            self.vision_model = MllamaForConditionalGeneration.from_pretrained(
                self.vlm,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            self.vision_processor = AutoProcessor.from_pretrained(self.vlm)
            print(f"✅ Llama Vision model loaded: {self.vision_model_name}")
            print(f"✅ Device: {next(self.vision_model.parameters()).device}")
            
        except Exception as e:
            print(f"❌ Error loading Llama Vision model: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you've accepted the Llama license at:")
            print("   https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct")
            print("2. Login with: huggingface-cli login")
            print("3. Check you have enough RAM/VRAM")
            raise

        self.vision_prompt = """Describe this scene in detail for a blind person who needs to navigate safely. Include:
            - All objects, furniture, and obstacles visible
            - Spatial layout (what's ahead, left, right)
            - Distances if possible (rough estimates in feet)
            - People or moving objects
            - Hazards like stairs, steps, or drop-offs
            - Doorways, walls, and openings

            Be specific, clear, and comprehensive."""

        self.navigation_prompt = """You are helping a blind person navigate based on this scene description:

            {description}

            Provide clear navigation guidance:
            1. Immediate hazards or obstacles to avoid
            2. Safe path directions (ahead, left, right)
            3. Distances and spatial awareness
            4. Specific next steps

            Keep response concise (2-3 sentences), actionable, speak directly to user ("you")."""


    def generate_scene_caption(self, frame):
        
        try:
            # Convert OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.vision_prompt}
                    ]
                }
            ]

            input_text = self.vision_processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            inputs = self.vision_processor(
                pil_img,
                input_text,
                return_tensors="pt"
            ).to(self.vision_model.device)

            # Generate description
            with torch.no_grad():
                output = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0
                )
            
            generated_text = self.vision_processor.decode(output[0], skip_special_tokens=False)

            # Get the assistants description
            if "assistant" in generated_text:
                description = generated_text.split("assistant")[-1].strip()
            else:
                description = generated_text.strip()
            
            return description
        
        except Exception as e:
            error_msg = f"Error with Llama Vision: {str(e)}"
            print(error_msg)
            return None
    
    def analyze_scene_with_cerebras(self, description):
        try:
            prompt = self.navigation_prompt.format(description=description)

            # Call Cerebras API
            response = self.client.chat.completions.create(
                model=self.cerebras_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful navigation assistant for blind people. Provide clear, concise, actionable guidance based on scene descriptions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=150
            )

            # Get the analysis
            guidance = response.choices[0].message.content.strip()
            return analysis

        except Exception as e:
            error_msg = f"Error with Cerebras analysis: {str(e)}"
            print(error_msg)
            return None
    
    def analyze(self, frame):
        try:
            # 1. Generate the image caption with Llama Vision
            print("[Llama-Vision] Analyzing scene....")
            caption = self.generate_scene_caption(frame)

            if caption is None:
                return None
            
            print(f"[Image Caption] {caption}")

            # 2. Analyze with Cerebras for proper guidance
            print("[Cerebras] Generating navigation guidance")
            guidance = self.analyze_scene_with_cerebras(caption)
            
            return guidance
        
        except Exception as e:
            error_msg = f"Error analyzing scene: {str(e)}"
            print(error_msg)
            return None
    

