#!/usr/bin/env python3
"""Use a GPT vision model to assign confidence scores to predicted boxes.

Input and output are RF20-style PKL files. The script keeps each prediction
box and category unchanged, and only updates the instance `score` field.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def require_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required for GPT confidence rescoring. Install with `python -m pip install -r requirements.txt`.") from exc
    return Image, ImageDraw, ImageFont


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pkl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a list, got {type(data).__name__}")
    return data


def save_pkl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(rows, f)


def clean_category_maps(coco: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
    has_dummy_none = bool(categories) and int(categories[0]["id"]) == 0 and str(categories[0].get("name", "")).lower() == "none"
    id_to_name: dict[int, str] = {}
    for category in categories:
        raw_id = int(category["id"])
        if has_dummy_none and raw_id == 0:
            continue
        cat_id = raw_id - 1 if has_dummy_none else raw_id
        id_to_name[cat_id] = str(category["name"])
    return id_to_name, {name: cat_id for cat_id, name in id_to_name.items()}


def pkl_by_image(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["image_id"]): row for row in rows if isinstance(row, dict) and "image_id" in row}


def clamp_bbox(bbox: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in [x, y, w, h]):
        return None
    x = max(0.0, min(float(width), x))
    y = max(0.0, min(float(height), y))
    w = max(0.0, min(float(width) - x, w))
    h = max(0.0, min(float(height) - y, h))
    return [x, y, w, h] if w > 0 and h > 0 else None


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    _, _, ImageFont = require_pillow()
    font = ImageFont.load_default()
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], fill=(0, 0, 0))
    draw.text(xy, text, fill=color, font=font)


def draw_box(image: Image.Image, bbox: list[float], label: str, color: tuple[int, int, int]) -> Image.Image:
    _, ImageDraw, _ = require_pillow()
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x, y, w, h = [int(round(value)) for value in bbox]
    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    draw_text(draw, (x + 2, max(2, y - 14)), label, color)
    return out


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def image_path(image_dir: Path, image: dict[str, Any]) -> Path:
    path = image_dir / str(image["file_name"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_confidence_prompt(dataset: str, label: str, bbox: list[float], width: int, height: int, use_exemplar: bool) -> str:
    if use_exemplar:
        image_desc = """You will receive two images:
- Image 1: a train example for this class. The green box shows the target appearance and box tightness.
- Image 2: the test image. The red box is the candidate prediction to score."""
        alignment = "is a correct detection of the target class and is well localized like the green example"
    else:
        image_desc = """You will receive one image:
- Image 1: the test image. The red box is the candidate prediction to score."""
        alignment = "is a correct detection of the target class and is tightly localized"

    return f"""You are judging one existing object-detection candidate box.

Dataset: {dataset}
Target class: {label}

{image_desc}

Candidate bbox in COCO pixels:
{json.dumps([round(float(value), 2) for value in bbox])}
Image size: width={width}, height={height}

Give a score from 0 to 1 for whether the red candidate box {alignment}.

Scoring guide:
- 1.00: correct class, clear target, tight box; IoU likely >= 0.75.
- 0.80: correct class, acceptable box; IoU likely 0.50-0.75.
- 0.60: probably correct class, but loose/shifted/partial.
- 0.40: uncertain target or weak overlap.
- 0.20: likely wrong class, wrong object, or very poor overlap.
- 0.00: no target of this class in the red box.

Return only JSON:
{{"score": 0.0}}"""


def parse_score_json(text: str) -> float | None:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        score = float(parsed.get("score"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return max(0.0, min(1.0, score))


def chat_completions_url(api_url: str, api_base: str) -> str:
    if api_url:
        return api_url
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def call_gpt_confidence(
    *,
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    images: list[Image.Image],
    timeout: int,
    retries: int,
) -> tuple[float | None, str, Any, str | None, float]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image), "detail": "high"}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict visual verifier for object-detection boxes. Return only JSON."},
            {"role": "user", "content": content},
        ],
        "max_tokens": 100,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    started = time.time()
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(2.0 * attempt)
                continue
            response.raise_for_status()
            raw = response.json()
            text = raw["choices"][0]["message"]["content"].strip()
            return parse_score_json(text), text, raw, None, time.time() - started
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(2.0 * attempt)
    return None, "", {}, last_error, time.time() - started


def auto_train_paths(coco_path: Path, image_dir: Path) -> tuple[Path | None, Path | None]:
    if coco_path.parent.name != "test":
        return None, None
    train_dir = coco_path.parent.parent / "train"
    train_coco = train_dir / "_annotations.coco.json"
    if train_coco.exists() and train_dir.exists():
        return train_coco, train_dir
    sibling_train = image_dir.parent / "train"
    sibling_coco = sibling_train / "_annotations.coco.json"
    if sibling_coco.exists() and sibling_train.exists():
        return sibling_coco, sibling_train
    return None, None


def make_train_exemplar(
    train_coco: dict[str, Any],
    train_image_dir: Path,
    cat_id: int,
    label: str,
) -> Image.Image | None:
    Image, _, _ = require_pillow()
    anns = [ann for ann in train_coco.get("annotations", []) if int(ann.get("category_id", -999)) == cat_id]
    if not anns:
        return None
    anns.sort(key=lambda ann: float(ann.get("area") or ann["bbox"][2] * ann["bbox"][3]))
    ann = anns[len(anns) // 2]
    image_by_id = {int(image["id"]): image for image in train_coco.get("images", [])}
    image = image_by_id.get(int(ann["image_id"]))
    if image is None:
        return None
    path = image_path(train_image_dir, image)
    img = Image.open(path).convert("RGB")
    bbox = clamp_bbox(ann.get("bbox"), int(image["width"]), int(image["height"]))
    if bbox is None:
        return img
    return draw_box(img, bbox, f"GT example: {label}", (0, 210, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use GPT vision scoring to update confidence scores in an RF20-style PKL.")
    parser.add_argument("--coco", required=True, type=Path, help="Test COCO _annotations.coco.json.")
    parser.add_argument("--image-dir", required=True, type=Path, help="Test image directory.")
    parser.add_argument("--input-pkl", required=True, type=Path, help="Input RF20-style PKL.")
    parser.add_argument("--output-pkl", required=True, type=Path, help="Output rescored PKL.")
    parser.add_argument("--dataset", default="", help="Dataset name used in the scoring prompt.")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"))
    parser.add_argument("--api-url", default=os.getenv("GPT_API_URL", os.getenv("OPENAI_CHAT_COMPLETIONS_URL", "")))
    parser.add_argument("--api-key", default=os.getenv("GPT_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    parser.add_argument("--model", default=os.getenv("GPT_MODEL", "gpt-5.4"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-boxes", type=int, default=0, help="Score only first N pending boxes. 0 means all.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Optional directory for raw GPT responses.")
    parser.add_argument("--use-train-exemplar", action="store_true", help="Include one green-box train exemplar before the red-box test image.")
    parser.add_argument("--train-coco", type=Path, default=None, help="Optional train COCO file for exemplars.")
    parser.add_argument("--train-image-dir", type=Path, default=None, help="Optional train image directory for exemplars.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output PKL scores.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned work without API calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key and not args.dry_run:
        raise SystemExit("Missing API key. Set GPT_API_KEY or OPENAI_API_KEY, or pass --api-key.")

    coco = read_json(args.coco)
    id_to_name, _ = clean_category_maps(coco)
    image_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    dataset = args.dataset or args.coco.parent.parent.name
    api_url = chat_completions_url(args.api_url, args.api_base)

    source_rows = load_pkl(args.input_pkl)
    rescored_rows = copy.deepcopy(source_rows)
    done_keys: set[tuple[int, int]] = set()
    if args.output_pkl.exists() and not args.no_resume:
        existing_by_image = pkl_by_image(load_pkl(args.output_pkl))
        for row in rescored_rows:
            image_id = int(row.get("image_id", -1))
            old_instances = row.get("instances", []) or []
            new_instances = existing_by_image.get(image_id, {}).get("instances", []) or []
            if len(old_instances) == len(new_instances):
                row["instances"] = new_instances
                done_keys.update((image_id, index) for index in range(len(new_instances)))

    train_coco = None
    train_image_dir = None
    if args.use_train_exemplar:
        train_coco_path = args.train_coco
        train_image_dir = args.train_image_dir
        if train_coco_path is None or train_image_dir is None:
            auto_coco, auto_dir = auto_train_paths(args.coco, args.image_dir)
            train_coco_path = train_coco_path or auto_coco
            train_image_dir = train_image_dir or auto_dir
        if train_coco_path is not None and train_image_dir is not None:
            train_coco = read_json(train_coco_path)

    exemplar_cache: dict[int, Image.Image | None] = {}
    if train_coco is not None and train_image_dir is not None and not args.dry_run:
        for cat_id, label in id_to_name.items():
            exemplar_cache[cat_id] = make_train_exemplar(train_coco, train_image_dir, cat_id, label)

    tasks: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for row_index, row in enumerate(rescored_rows):
        image_id = int(row.get("image_id", -1))
        image = image_by_id.get(image_id)
        if image is None:
            continue
        for inst_index, instance in enumerate(row.get("instances", []) or []):
            if (image_id, inst_index) in done_keys:
                continue
            try:
                cat_id = int(instance["category_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if cat_id not in id_to_name:
                continue
            tasks.append((row_index, inst_index, image, instance))

    if args.max_boxes > 0:
        tasks = tasks[: args.max_boxes]

    print(f"Dataset: {dataset}")
    print(f"Input PKL: {args.input_pkl}")
    print(f"Output PKL: {args.output_pkl}")
    print(f"Model: {args.model}")
    print(f"API URL: {api_url}")
    print(f"Boxes already scored: {len(done_keys)}")
    print(f"Boxes pending: {len(tasks)}")
    print(f"Train exemplars: {'enabled' if exemplar_cache else 'disabled'}")
    if args.dry_run:
        return 0

    failures: list[dict[str, Any]] = []

    def score_one(task: tuple[int, int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        row_index, inst_index, image, instance = task
        image_id = int(image["id"])
        cat_id = int(instance["category_id"])
        label = id_to_name.get(cat_id, str(cat_id))
        bbox = clamp_bbox(instance.get("bbox"), int(image["width"]), int(image["height"]))
        if bbox is None:
            return {"row_index": row_index, "inst_index": inst_index, "image_id": image_id, "score": None, "error": "invalid bbox"}

        Image, _, _ = require_pillow()
        original = Image.open(image_path(args.image_dir, image)).convert("RGB")
        candidate = draw_box(original, bbox, f"{label}", (230, 30, 30))
        images = []
        exemplar = exemplar_cache.get(cat_id)
        if exemplar is not None:
            images.append(exemplar)
        images.append(candidate)
        prompt = build_confidence_prompt(dataset, label, bbox, int(image["width"]), int(image["height"]), exemplar is not None)
        score, text, raw, error, seconds = call_gpt_confidence(
            api_url=api_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            images=images,
            timeout=args.timeout,
            retries=args.retries,
        )
        return {
            "row_index": row_index,
            "inst_index": inst_index,
            "image_id": image_id,
            "label": label,
            "old_score": instance.get("score"),
            "score": score,
            "duration_s": seconds,
            "error": error if error else (None if score is not None else "parse_score_failed"),
            "raw_text": text,
            "raw_response": raw,
        }

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(score_one, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            done += 1
            row_index = int(result["row_index"])
            inst_index = int(result["inst_index"])
            image_id = int(result["image_id"])
            if result.get("score") is not None:
                rescored_rows[row_index]["instances"][inst_index]["score"] = float(result["score"])
            else:
                failures.append({"image_id": image_id, "inst_index": inst_index, "error": result.get("error")})

            if args.raw_dir is not None:
                write_json(args.raw_dir / f"{image_id}_{inst_index}.json", result)
            save_pkl(args.output_pkl, rescored_rows)
            status = "failed" if result.get("error") else "ok"
            print(f"[{done}/{len(tasks)}] {status} image_id={image_id} box={inst_index} score={result.get('score')}", flush=True)

    save_pkl(args.output_pkl, rescored_rows)
    if args.raw_dir is not None and failures:
        write_json(args.raw_dir / "failures.json", failures)
    print(f"Saved: {args.output_pkl}")
    print(f"Failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
