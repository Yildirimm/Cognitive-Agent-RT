import os
from typing import Dict
from google import genai
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")


client = genai.Client(api_key=API_KEY)


MODEL_NAME = "models/gemini-flash-latest"

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["forward", "turn_left", "turn_right", "stop"],
        }
    },
    "required": ["action"],
}


def gemini_llm_client(prompt: Dict[str, str]) -> str:
    """
    Returns a JSON string that follows ACTION_SCHEMA
    """


    full_prompt = (
        f"{prompt['system']}\n\n"
        f"{prompt['user']}"
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt,
        config={
            "max_output_tokens": 512,
            "response_mime_type": "application/json",
            "response_schema": ACTION_SCHEMA,
            },
            )
    
    # Structured output is already JSON-safe
    if not response.candidates:
        return "No response from Gemini"
    

    content = response.candidates[0].content
    if not content or not content.parts:
        return "No response from Gemini"
        

    return content.parts[0].text.strip()
