#!/usr/bin/env python3
"""Replace or add selected categories from one RF20-style PKL into another."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


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


def load_category_names(coco_path: Path | None) -> dict[str, int]:
    if coco_path is None:
        return {}
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
    has_dummy_none = bool(categories) and int(categories[0]["id"]) == 0 and str(categories[0].get("name", "")).lower() == "none"
    out: dict[str, int] = {}
    for cat in categories:
        raw_id = int(cat["id"])
        if has_dummy_none and raw_id == 0:
            continue
        out[str(cat["name"])] = raw_id - 1 if has_dummy_none else raw_id
    return out


def parse_categories(values: list[str], name_to_id: dict[str, int]) -> set[int]:
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
    if not out:
        raise ValueError("At least one category must be provided")
    return out


def index_by_image(rows: list[dict[str, Any]], label: str) -> dict[Any, dict[str, Any]]:
    by_image: dict[Any, dict[str, Any]] = {}
    for row in rows:
        image_id = row.get("image_id")
        if image_id in by_image:
            raise ValueError(f"{label} has duplicate image_id={image_id}")
        by_image[image_id] = row
    return by_image


def filtered_instances(row: dict[str, Any], category_ids: set[int], keep_selected: bool) -> list[dict[str, Any]]:
    out = []
    for instance in row.get("instances", []) or []:
        try:
            selected = int(instance.get("category_id")) in category_ids
        except (TypeError, ValueError):
            selected = False
        if selected == keep_selected:
            out.append(dict(instance))
    return out


def merge_categories(
    base_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    *,
    category_ids: set[int],
    mode: str,
    require_all_images: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_by_image = index_by_image(source_rows, "source")
    base_ids = [row.get("image_id") for row in base_rows]
    source_ids = set(source_by_image)
    missing = sorted(set(base_ids) - source_ids)
    if require_all_images and missing:
        raise ValueError(f"Source PKL is missing image ids: {missing[:20]}")

    output: list[dict[str, Any]] = []
    stats = {
        "images": len(base_rows),
        "removed_base_instances": 0,
        "kept_base_instances": 0,
        "inserted_source_instances": 0,
        "output_instances": 0,
        "source_extra_images": len(source_ids - set(base_ids)),
    }

    for base_row in base_rows:
        image_id = base_row.get("image_id")
        source_row = source_by_image.get(image_id, {"instances": []})
        base_selected = filtered_instances(base_row, category_ids, keep_selected=True)
        base_other = filtered_instances(base_row, category_ids, keep_selected=False)
        source_selected = filtered_instances(source_row, category_ids, keep_selected=True)

        if mode == "replace":
            instances = base_other + source_selected
            stats["removed_base_instances"] += len(base_selected)
        elif mode == "append":
            instances = list(base_row.get("instances", []) or []) + source_selected
        elif mode == "only":
            instances = source_selected
            stats["removed_base_instances"] += len(base_row.get("instances", []) or [])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        fixed_instances = []
        for instance in instances:
            fixed = dict(instance)
            fixed["image_id"] = image_id
            fixed_instances.append(fixed)

        output.append({"image_id": image_id, "instances": fixed_instances})
        stats["kept_base_instances"] += len(base_other) if mode == "replace" else len(base_row.get("instances", []) or [])
        stats["inserted_source_instances"] += len(source_selected)
        stats["output_instances"] += len(fixed_instances)

    return output, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge selected category predictions from source PKL into base PKL.")
    parser.add_argument("--base", required=True, type=Path, help="Base RF20-style PKL.")
    parser.add_argument("--source", required=True, type=Path, help="Source RF20-style PKL.")
    parser.add_argument("--output", required=True, type=Path, help="Output merged PKL.")
    parser.add_argument("--coco", type=Path, default=None, help="Optional COCO annotation file, used to resolve category names.")
    parser.add_argument("--category", action="append", required=True, help="Category id/name. Can be repeated or comma-separated.")
    parser.add_argument("--mode", choices=["replace", "append", "only"], default="replace")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name_to_id = load_category_names(args.coco)
    category_ids = parse_categories(args.category, name_to_id)
    merged, stats = merge_categories(
        load_pkl(args.base),
        load_pkl(args.source),
        category_ids=category_ids,
        mode=args.mode,
        require_all_images=not args.allow_missing_images,
    )
    print(f"Base: {args.base}")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    print(f"Categories: {sorted(category_ids)}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    if not args.dry_run:
        save_pkl(args.output, merged)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
