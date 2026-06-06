from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image


def call_gemini_image_model(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    image_bytes: bytes,
    mime: str,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    seed: int | None = None,
    timeout: int = 240,
) -> tuple[list[Image.Image], str, Any, str | None, float]:
    if not api_key:
        return [], "", {}, "Missing GEMINI_API_KEY", 0.0

    generation_config: dict[str, Any] = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_output_tokens),
    }
    if seed is not None:
        generation_config["seed"] = int(seed)

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime, "data": base64.b64encode(image_bytes).decode("utf-8")}},
            ],
        }],
        "generationConfig": generation_config,
    }
    url = f"{api_base.rstrip('/')}/v1beta/models/{model}:generateContent"
    t0 = time.time()
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()
        images: list[Image.Image] = []
        texts: list[str] = []
        for cand in raw.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part and not part.get("thought"):
                    texts.append(part["text"])
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    data = base64.b64decode(inline["data"])
                    images.append(Image.open(io.BytesIO(data)).convert("RGB"))
        return images, "\n".join(texts).strip(), raw, None, time.time() - t0
    except Exception as exc:
        return [], "", {}, str(exc), time.time() - t0


def image_mime(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
