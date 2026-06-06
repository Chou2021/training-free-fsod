from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any


COLOR_PRESETS: dict[str, dict[str, str]] = {
    "all6": {
        "red": "DIP",
        "blue": "PIP",
        "yellow": "MCP",
        "green": "Radius",
        "orange": "Ulna",
        "purple": "Wrist",
    },
    "joints": {
        "red": "DIP",
        "blue": "PIP",
        "yellow": "MCP",
    },
    "wrist_forearm": {
        "green": "Radius",
        "orange": "Ulna",
        "purple": "Wrist",
    },
    "forearm_bones": {
        "green": "Radius",
        "orange": "Ulna",
    },
    "dentalai": {
        "red": "Cavity",
        "blue": "Fillings",
        "yellow": "Impacted Tooth",
        "green": "Implant",
    },
}


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("._") or "run"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_pkl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(rows, f)


def load_coco(coco_path: Path) -> dict[str, Any]:
    coco = read_json(coco_path)
    if "images" not in coco or "categories" not in coco:
        raise ValueError(f"Invalid COCO file: {coco_path}")
    return coco


def category_maps(coco: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name = {int(cat["id"]): str(cat["name"]) for cat in coco.get("categories", [])}
    return id_to_name, {v: k for k, v in id_to_name.items()}


def select_images(coco: dict[str, Any], first_n: int = 0, image_ids: set[int] | None = None) -> list[dict[str, Any]]:
    images = sorted(coco["images"], key=lambda x: int(x["id"]))
    if image_ids is not None:
        images = [img for img in images if int(img["id"]) in image_ids]
    if first_n and first_n > 0:
        images = images[:first_n]
    return images


def image_path(image_dir: Path, image: dict[str, Any]) -> Path:
    path = image_dir / str(image["file_name"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path
