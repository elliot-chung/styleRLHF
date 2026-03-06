"""VLM judge: send composite outfit images to GPT-4o or Gemini and get preference A/B."""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Literal, Optional
from pydantic import BaseModel, Field
from enum import StrEnum
from .config import GEMMA_RPM_LIMIT

from PIL import Image

class Winner(StrEnum):
    A = "A"
    B = "B"

class JudgeResponse(BaseModel):
    explanation: str = Field(description="The explanation for the winner.")
    winner: Winner = Field(description="The winner of the comparison.")

def image_to_base64_jpeg(pil_image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def call_openai_judge(
    image_a: Image.Image,
    image_b: Image.Image,
    prompt: str,
    model: str = "gpt-4o",
) -> Literal["A", "B"]:
    """Use OpenAI GPT-4o to compare two outfit images. Returns 'A' or 'B'."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    b64_a = image_to_base64_jpeg(image_a)
    b64_b = image_to_base64_jpeg(image_b)
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Outfit A:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_a}", "detail": "low"},
        },
        {"type": "text", "text": "Outfit B:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_b}", "detail": "low"},
        },
    ]
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], max_tokens=200)
    text = (resp.choices[0].message.content or "").strip().upper()
    # Structured response: {"winner": "A", "explanation": "..."}

# Gemma is cheaper but doesn't support structured outputs (requires regex parsing)    
def call_gemma_judge(
    image_a: Image.Image,
    image_b: Image.Image,
    prompt: str,
) -> Literal["A", "B"]:
    """Use Google Gemma to compare two outfit images. Returns 'A' or 'B'."""
    try:
        from google import genai
    except ImportError:
        raise ImportError("Install google-genai: pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    model_name = "gemma-3-27b-it"
    content = [
        prompt, 
        "Outfit A:", image_a, 
        "Outfit B:", image_b,
    ]
    
    resp = client.models.generate_content(model=model_name, contents=content)

    text = resp.text.strip().upper()
    
    result = "A" if text.rfind("A") > text.rfind("B") else "B"
    
    return result

def call_gemini_judge(
    image_a: Image.Image,
    image_b: Image.Image,
    prompt: str,
) -> Literal["A", "B"]:
    """Use Google Gemini to compare two outfit images. Returns 'A' or 'B'."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Install google-genai: pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    model_name = "gemini-3-flash-preview"
    content = [
        prompt, 
        "Outfit A:", image_a, 
        "Outfit B:", image_b,
    ]
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JudgeResponse.model_json_schema()
    )
    resp = client.models.generate_content(model=model_name, contents=content, config=config)
    
    resp = JudgeResponse.model_validate_json(resp.text)
    
    return resp.winner


def judge_pair(
    image_a: Image.Image,
    image_b: Image.Image,
    prompt: Optional[str] = None,
    backend: Literal["openai", "gemini", "gemma"] = "gemma",
) -> Literal["A", "B"]:
    """Unified interface: compare two outfit images and return preferred (A or B)."""
    from .config import VLM_JUDGE_PROMPT
    prompt = prompt or VLM_JUDGE_PROMPT
    if backend == "gemma":
        return call_gemma_judge(image_a, image_b, prompt)
    elif backend == "gemini":
        return call_gemini_judge(image_a, image_b, prompt)
    elif backend == "openai":
        return call_openai_judge(image_a, image_b, prompt)
    else:
        raise ValueError(f"Invalid backend: {backend}")
