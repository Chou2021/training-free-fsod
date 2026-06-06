from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from xray_gen_bbox.common import save_pkl
from xray_gen_bbox.extract_boxes import bbox_iou


def load_pkl(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else []


def dedupe(instances: list[dict[str, Any]], iou_thr: float) -> list[dict[str, Any]]:
    if iou_thr <= 0:
        return instances
    kept = []
    for inst in instances:
        duplicate = False
        for old in kept:
            if int(inst.get("category_id", -1)) == int(old.get("category_id", -2)) and bbox_iou(inst.get("bbox"), old.get("bbox")) >= iou_thr:
                duplicate = True
                break
        if not duplicate:
            kept.append(inst)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple RF20-style pkl files by image_id.")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out-pkl", required=True, type=Path)
    parser.add_argument("--dedupe-iou", type=float, default=0.8)
    args = parser.parse_args()

    merged: dict[int, dict[str, Any]] = {}
    for path in args.inputs:
        for row in load_pkl(path):
            image_id = int(row.get("image_id", -1))
            if image_id < 0:
                continue
            merged.setdefault(image_id, {"image_id": image_id, "instances": []})
            merged[image_id]["instances"].extend(row.get("instances", []) or [])
    rows = []
    for image_id in sorted(merged):
        row = merged[image_id]
        row["instances"] = dedupe(row.get("instances", []) or [], args.dedupe_iou)
        rows.append(row)
    save_pkl(args.out_pkl, rows)
    print(f"Wrote {args.out_pkl} | images={len(rows)} | boxes={sum(len(r.get('instances', []) or []) for r in rows)}")


if __name__ == "__main__":
    main()
