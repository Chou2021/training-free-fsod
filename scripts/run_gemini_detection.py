#!/usr/bin/env python3
"""Run a Gemini-style vision model on a COCO split and export RF20-style PKL.

The model is expected to return JSON detections with Gemini-native boxes:
[
  {"label": "class name", "box_2d": [ymin, xmin, ymax, xmax], "confidence": 1}
]

Coordinates are normalized to [0, 1000]. This script converts them to COCO
[x, y, width, height] pixel boxes and writes:
[
  {"image_id": 0, "instances": [{"image_id": 0, "category_id": 1, ...}]}
]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import pickle
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pkl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a list, got {type(data).__name__}")
    return data


def save_pkl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(rows, f)


def clean_coco_categories(coco: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    """Return category maps, dropping Roboflow dummy 'none' when present.

    Some RF20-VL local COCO files include category id 0 named "none" while the
    official submission expects real classes to start at 0. The competition
    data used here generally does not have this dummy in test annotations, but
    keeping the check makes the runner safer for Roboflow exports.
    """

    categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
    has_dummy_none = bool(categories) and int(categories[0]["id"]) == 0 and str(categories[0].get("name", "")).lower() == "none"
    id_to_name: dict[int, str] = {}
    for cat in categories:
        raw_id = int(cat["id"])
        name = str(cat["name"])
        if has_dummy_none and raw_id == 0:
            continue
        cat_id = raw_id - 1 if has_dummy_none else raw_id
        id_to_name[cat_id] = name
    return id_to_name, {name: cat_id for cat_id, name in id_to_name.items()}


def parse_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def select_images(coco: dict[str, Any], first_n: int = 0, image_ids: set[int] | None = None) -> list[dict[str, Any]]:
    images = sorted(coco.get("images", []), key=lambda image: int(image["id"]))
    if image_ids is not None:
        images = [image for image in images if int(image["id"]) in image_ids]
    if first_n > 0:
        images = images[:first_n]
    return images


def mime_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    if path.suffix.lower() == ".png":
        return "image/png"
    if path.suffix.lower() == ".webp":
        return "image/webp"
    return "image/jpeg"


def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
    return text.strip()


def parse_jsonish(text: str) -> Any:
    text = strip_json_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if not match:
            return []
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return []


def clamp_bbox(bbox: Any, width: int, height: int, min_size: float = 0.0) -> list[float] | None:
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
    if w <= min_size or h <= min_size:
        return None
    return [x, y, w, h]


def box2d_to_coco(box: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [float(value) for value in box[:4]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in [ymin, xmin, ymax, xmax]):
        return None
    ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, value)) for value in [ymin, xmin, ymax, xmax]]
    if ymax <= ymin or xmax <= xmin:
        return None
    return clamp_bbox(
        [
            xmin / 1000.0 * width,
            ymin / 1000.0 * height,
            (xmax - xmin) / 1000.0 * width,
            (ymax - ymin) / 1000.0 * height,
        ],
        width,
        height,
        min_size=1.0,
    )


def score_from_item(item: dict[str, Any]) -> float:
    raw = item.get("score", item.get("confidence", 1.0))
    try:
        score = float(raw)
    except (TypeError, ValueError):
        score = 1.0
    if score > 1.0:
        score = score / 5.0
    return max(0.0, min(1.0, score))


def normalize_label(label: Any) -> str:
    return str(label).strip() if label is not None else ""


def resolve_category_id(label: str, item: dict[str, Any], name_to_id: dict[str, int], label_aliases: dict[str, str]) -> int | None:
    label = label_aliases.get(label, label)
    if label in name_to_id:
        return int(name_to_id[label])
    lowered = {name.lower(): cat_id for name, cat_id in name_to_id.items()}
    if label.lower() in lowered:
        return int(lowered[label.lower()])
    if "category_id" in item:
        try:
            return int(item["category_id"])
        except (TypeError, ValueError):
            return None
    return None


def parse_instances(
    text: str,
    *,
    image_id: int,
    width: int,
    height: int,
    name_to_id: dict[str, int],
    label_aliases: dict[str, str],
    allowed_category_ids: set[int] | None,
) -> tuple[list[dict[str, Any]], Any]:
    parsed = parse_jsonish(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("detections", parsed.get("results", parsed.get("instances", parsed)))
    if not isinstance(parsed, list):
        return [], parsed

    instances: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        label = normalize_label(item.get("label") or item.get("class") or item.get("class_name") or item.get("name"))
        cat_id = resolve_category_id(label, item, name_to_id, label_aliases)
        if cat_id is None:
            continue
        if allowed_category_ids is not None and cat_id not in allowed_category_ids:
            continue

        bbox = None
        if any(key in item for key in ("box_2d", "box2d", "bbox_2d")):
            bbox = box2d_to_coco(item.get("box_2d") or item.get("box2d") or item.get("bbox_2d"), width, height)
        elif "bbox" in item:
            bbox = clamp_bbox(item["bbox"], width, height, min_size=1.0)
        if bbox is None:
            continue

        instances.append(
            {
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "score": score_from_item(item),
            }
        )
    return instances, parsed


def extract_text(raw_response: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in raw_response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part and not part.get("thought"):
                texts.append(str(part["text"]))
    return "\n".join(texts).strip()


def build_api_url(api_base: str, model: str, api_url: str | None) -> str:
    if api_url:
        return api_url
    return f"{api_base.rstrip('/')}/v1beta/models/{model}:generateContent"


def call_gemini(
    *,
    api_url: str,
    api_key: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type_for(image_path),
                            "data": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def error_body(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return f"HTTP {exc.code}: {body or exc.reason}"
    return str(exc)


def format_prompt(template: str, dataset: str, image: dict[str, Any], class_names: list[str]) -> str:
    return (
        template.replace("{dataset}", dataset)
        .replace("{image_id}", str(image.get("id", "")))
        .replace("{file_name}", str(image.get("file_name", "")))
        .replace("{width}", str(int(image.get("width", 0))))
        .replace("{height}", str(int(image.get("height", 0))))
        .replace("{class_names}", ", ".join(class_names))
    )


def load_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    aliases = read_json(path)
    if not isinstance(aliases, dict):
        raise TypeError("--label-aliases must point to a JSON object")
    return {str(k): str(v) for k, v in aliases.items()}


def parse_category_filter(values: list[str] | None, name_to_id: dict[str, int]) -> set[int] | None:
    if not values:
        return None
    out: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if part in name_to_id:
                out.add(int(name_to_id[part]))
            else:
                out.add(int(part))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini detection over a COCO split and write an RF20-style PKL.")
    parser.add_argument("--coco", required=True, type=Path, help="Path to _annotations.coco.json.")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing images referenced by COCO.")
    parser.add_argument("--prompt", required=True, type=Path, help="Prompt text file.")
    parser.add_argument("--output", required=True, type=Path, help="Output PKL path.")
    parser.add_argument("--dataset", default="", help="Dataset name for prompt placeholders. Defaults to COCO parent name.")
    parser.add_argument("--api-base", default=os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com"))
    parser.add_argument("--api-url", default=os.getenv("GEMINI_API_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--first-n", type=int, default=0, help="Run only first N selected images. 0 means all.")
    parser.add_argument("--image-ids", default="", help="Comma-separated COCO image ids.")
    parser.add_argument("--category", action="append", help="Optional category name/id filter. Can be repeated or comma-separated.")
    parser.add_argument("--label-aliases", type=Path, default=None, help="Optional JSON mapping model labels to COCO category names.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory for raw model responses and parse logs.")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse image records already present in --output.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned work without API calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key and not args.dry_run:
        print("Missing API key. Set GEMINI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    coco = read_json(args.coco)
    id_to_name, name_to_id = clean_coco_categories(coco)
    class_names = [id_to_name[cat_id] for cat_id in sorted(id_to_name)]
    allowed_category_ids = parse_category_filter(args.category, name_to_id)
    label_aliases = load_aliases(args.label_aliases)
    dataset = args.dataset or args.coco.parent.parent.name
    images = select_images(coco, first_n=args.first_n, image_ids=parse_ids(args.image_ids))
    prompt_template = args.prompt.read_text(encoding="utf-8").strip()
    api_url = build_api_url(args.api_base, args.model, args.api_url or None)

    existing_by_image: dict[int, dict[str, Any]] = {}
    if args.output.exists() and not args.no_resume:
        for row in load_pkl(args.output):
            if isinstance(row, dict) and "image_id" in row:
                existing_by_image[int(row["image_id"])] = row

    selected_ids = [int(image["id"]) for image in images]
    pending = [image for image in images if int(image["id"]) not in existing_by_image]

    print(f"Dataset: {dataset}")
    print(f"COCO: {args.coco}")
    print(f"Image dir: {args.image_dir}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print(f"API URL: {api_url}")
    print(f"Model: {args.model}")
    print(f"Images selected: {len(images)}; resumed: {len(existing_by_image)}; pending: {len(pending)}")
    if allowed_category_ids is not None:
        print("Category filter:", ", ".join(f"{cat_id}:{id_to_name.get(cat_id, '?')}" for cat_id in sorted(allowed_category_ids)))
    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.raw_dir is not None:
        args.raw_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, Any]] = []

    def run_one(image: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        image_id = int(image["id"])
        img_path = args.image_dir / str(image["file_name"])
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        prompt = format_prompt(prompt_template, dataset, image, class_names)
        raw_response: dict[str, Any] | None = None
        error: str | None = None
        started = time.time()
        for attempt in range(1, args.retries + 1):
            try:
                raw_response = call_gemini(
                    api_url=api_url,
                    api_key=args.api_key,
                    prompt=prompt,
                    image_path=img_path,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.timeout,
                )
                error = None
                break
            except Exception as exc:
                error = error_body(exc)
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)
        seconds = time.time() - started
        if raw_response is None:
            record = {"image_id": image_id, "instances": []}
            raw_log = {"image_id": image_id, "file_name": image["file_name"], "duration_s": seconds, "error": error}
            return image_id, record, raw_log

        text = extract_text(raw_response)
        instances, parsed = parse_instances(
            text,
            image_id=image_id,
            width=int(image["width"]),
            height=int(image["height"]),
            name_to_id=name_to_id,
            label_aliases=label_aliases,
            allowed_category_ids=allowed_category_ids,
        )
        record = {"image_id": image_id, "instances": instances}
        raw_log = {
            "image_id": image_id,
            "file_name": image["file_name"],
            "duration_s": seconds,
            "error": None,
            "raw_text": text,
            "parsed": parsed,
            "n_boxes": len(instances),
            "raw_response": raw_response,
        }
        return image_id, record, raw_log

    completed = dict(existing_by_image)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_to_image = {executor.submit(run_one, image): image for image in pending}
        for done_count, future in enumerate(as_completed(future_to_image), start=1):
            image = future_to_image[future]
            image_id = int(image["id"])
            try:
                returned_id, record, raw_log = future.result()
                completed[returned_id] = record
                if raw_log.get("error"):
                    failures.append({"image_id": returned_id, "file_name": image["file_name"], "error": raw_log["error"]})
            except Exception as exc:
                failures.append({"image_id": image_id, "file_name": image["file_name"], "error": error_body(exc)})
                raw_log = {"image_id": image_id, "file_name": image["file_name"], "error": error_body(exc)}
                completed[image_id] = {"image_id": image_id, "instances": []}

            if args.raw_dir is not None:
                write_json(args.raw_dir / f"{image_id}.json", raw_log)
            rows = [completed[image_id] for image_id in selected_ids if image_id in completed]
            save_pkl(args.output, rows)
            n_boxes = sum(len(row.get("instances", []) or []) for row in rows)
            status = "failed" if raw_log.get("error") else "ok"
            print(f"[{done_count}/{len(pending)}] {status} image_id={image_id} boxes={len(completed[image_id].get('instances', []))} total_boxes={n_boxes}", flush=True)

    rows = [completed[image_id] for image_id in selected_ids if image_id in completed]
    save_pkl(args.output, rows)
    print(f"Wrote {args.output} | images={len(rows)} | boxes={sum(len(row.get('instances', []) or []) for row in rows)}")
    if failures:
        if args.raw_dir is not None:
            write_json(args.raw_dir / "failures.json", failures)
        print(f"Finished with {len(failures)} failed images. Re-run the same command to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
