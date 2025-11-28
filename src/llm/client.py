import os
import toml
from typing import Optional, Dict, Any
import openai

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            # Try to load from config/settings.toml
            try:
                # Assuming run from root
                config_path = os.path.join(os.getcwd(), "config", "settings.toml")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = toml.load(f)
                        self.api_key = config.get("openai", {}).get("api_key")
            except Exception as e:
                print(f"Warning: Failed to load OpenAI API key from settings.toml: {e}")

        if not self.api_key:
            # Fallback for development/testing if no key is present, 
            # but warn heavily or mock if needed. 
            # For now, we'll just print a warning and let the call fail if used.
            print("Warning: OpenAI API key not found. LLM features will fail.")

        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
            
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.client:
            return "Error: LLM Client not initialized with API Key."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
