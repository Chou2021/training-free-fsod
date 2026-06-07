from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image


def image_mime(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def call_gemini_image(
    image_path: Path,
    prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
) -> tuple[Image.Image | None, dict[str, Any], str | None, float]:
    url = f"{api_base.rstrip('/')}/v1beta/models/{model}:generateContent"
    mime = image_mime(image_path)
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
                    }
                },
            ],
        }],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    t0 = time.time()
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload,
            timeout=240,
        )
        resp.raise_for_status()
        raw = resp.json()
        for cand in raw.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    data = base64.b64decode(inline["data"])
                    return Image.open(io.BytesIO(data)).convert("RGB"), raw, None, time.time() - t0
        return None, raw, "No image returned by model", time.time() - t0
    except Exception as exc:
        return None, {}, str(exc), time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Gemini image model to draw colored x-ray detection boxes.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--out-image", required=True, type=Path)
    parser.add_argument("--raw-json", type=Path, default=None)
    parser.add_argument("--api-base", default=os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com"))
    parser.add_argument("--model", default=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"))
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set GEMINI_API_KEY or pass --api-key.")

    prompt = args.prompt.read_text(encoding="utf-8")
    image, raw, error, seconds = call_gemini_image(
        args.image,
        prompt,
        args.api_key,
        args.api_base,
        args.model,
        args.temperature,
    )
    if args.raw_json:
        args.raw_json.parent.mkdir(parents=True, exist_ok=True)
        args.raw_json.write_text(
            json.dumps({"error": error, "duration_s": seconds, "raw": raw}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if error or image is None:
        raise SystemExit(f"Model call failed: {error}")

    args.out_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out_image)
    print(f"Saved annotated image: {args.out_image}")
    print(f"Duration: {seconds:.2f}s")


if __name__ == "__main__":
    main()
