"""Vision client — Gemini 3.1 Flash Lite with key rotation (500 req/day per key)."""
import os
import base64
import asyncio
import logging
import random
from PIL import Image
import io as io_module

logger = logging.getLogger(__name__)

# All Gemini API keys — 500 requests/day per key on gemini-3.1-flash-lite
_GEMINI_KEYS = [
    k for k in [
        os.environ.get("GOOGLE_AI_API_KEY", ""),
        os.environ.get("GOOGLE_AI_API_KEY_2", ""),
        os.environ.get("GOOGLE_AI_API_KEY_3", ""),
        os.environ.get("GOOGLE_AI_API_KEY_4", ""),
    ] if k
]

if not _GEMINI_KEYS:
    raise ValueError("No GOOGLE_AI_API_KEY keys configured")

# Use Gemini 3.1 Flash Lite — 500 RPD, 15 RPM, vision-capable
VISION_MODEL = "models/gemini-3.1-flash-lite"

async def vision_call(
    prompt: str,
    image_bytes: bytes,
    max_tokens: int = 400,
    temperature: float = 0.1,
) -> str:
    """Vision call with automatic Gemini key rotation — 500 req/day per key."""
    import google.generativeai as genai
    
    keys = _GEMINI_KEYS.copy()
    random.shuffle(keys)
    
    for key in keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(VISION_MODEL)
            image = Image.open(io_module.BytesIO(image_bytes))
            
            response = await asyncio.to_thread(
                lambda: model.generate_content([prompt, image])
            )
            return response.text
            
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "exceeded" in err:
                logger.warning(f"Gemini key exhausted — rotating to next")
                continue
            raise
    
    raise Exception("All Gemini API keys exhausted")