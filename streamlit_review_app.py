"""Lightweight RF20-VL prediction viewer.

Run:
    streamlit run streamlit_review_app.py
"""

from __future__ import annotations

import io
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pkl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a list, got {type(data).__name__}")
    return data


def clean_category_maps(coco: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
    has_dummy_none = bool(categories) and int(categories[0]["id"]) == 0 and str(categories[0].get("name", "")).lower() == "none"
    id_to_name: dict[int, str] = {}
    for cat in categories:
        raw_id = int(cat["id"])
        if has_dummy_none and raw_id == 0:
            continue
        cat_id = raw_id - 1 if has_dummy_none else raw_id
        id_to_name[cat_id] = str(cat["name"])
    return id_to_name, {name: cat_id for cat_id, name in id_to_name.items()}


def clamp_bbox(bbox: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in [x, y, w, h]):
        return None
    x = max(0.0, min(float(width), x))
    y = max(0.0, min(float(height), y))
    w = max(0.0, min(float(width) - x, w))
    h = max(0.0, min(float(height) - y, h))
    return [x, y, w, h] if w > 0 and h > 0 else None


def bbox_iou(a: Any, b: Any) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def predictions_by_image(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    return {int(row["image_id"]): row.get("instances", []) or [] for row in rows if isinstance(row, dict) and "image_id" in row}


def annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        out[int(ann["image_id"])].append(ann)
    return dict(out)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], fill=(0, 0, 0))
    draw.text(xy, text, fill=color, font=font)


def draw_box(draw: ImageDraw.ImageDraw, bbox: list[float], label: str, color: tuple[int, int, int]) -> None:
    x, y, w, h = [int(round(v)) for v in bbox]
    draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
    draw_label(draw, (x + 2, max(2, y - 14)), label, color)


def render_overlay(
    image_path: Path,
    gt_anns: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    id_to_name: dict[int, str],
    mode: str,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if mode in {"gt", "both"}:
        for ann in gt_anns:
            bbox = clamp_bbox(ann.get("bbox"), width, height)
            if bbox is None:
                continue
            label = id_to_name.get(int(ann.get("category_id", -1)), str(ann.get("category_id", "")))
            draw_box(draw, bbox, f"GT {label}", (0, 200, 0))
    if mode in {"pred", "both"}:
        for index, inst in enumerate(pred_instances):
            bbox = clamp_bbox(inst.get("bbox"), width, height)
            if bbox is None:
                continue
            cat_id = int(inst.get("category_id", -1))
            label = id_to_name.get(cat_id, str(cat_id))
            score = float(inst.get("score", 0.0))
            draw_box(draw, bbox, f"P{index} {label} {score:.2f}", (230, 30, 30))
    return image


def image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def image_summary(
    image: dict[str, Any],
    gt_anns: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
) -> dict[str, Any]:
    width = int(image["width"])
    height = int(image["height"])
    gt_by_cat: dict[int, list[list[float]]] = defaultdict(list)
    for ann in gt_anns:
        bbox = clamp_bbox(ann.get("bbox"), width, height)
        if bbox is not None:
            gt_by_cat[int(ann["category_id"])].append(bbox)

    tp50 = 0
    low_iou = 0
    no_same_class_gt = 0
    matched: dict[int, set[int]] = defaultdict(set)
    for inst in pred_instances:
        bbox = clamp_bbox(inst.get("bbox"), width, height)
        if bbox is None:
            continue
        cat_id = int(inst.get("category_id", -1))
        candidates = gt_by_cat.get(cat_id, [])
        best_iou, best_index = 0.0, -1
        for index, gt_box in enumerate(candidates):
            if index in matched[cat_id]:
                continue
            iou = bbox_iou(bbox, gt_box)
            if iou > best_iou:
                best_iou, best_index = iou, index
        if not candidates:
            no_same_class_gt += 1
        elif best_iou >= 0.5 and best_index >= 0:
            tp50 += 1
            matched[cat_id].add(best_index)
        else:
            low_iou += 1
    return {
        "image_id": int(image["id"]),
        "file_name": image["file_name"],
        "n_gt": len(gt_anns),
        "n_pred": len(pred_instances),
        "tp50": tp50,
        "possible_missing_gt": max(0, len(gt_anns) - tp50),
        "low_iou_or_wrong_class": low_iou + no_same_class_gt,
    }


def main() -> None:
    st.set_page_config(page_title="RF20-VL Prediction Review", layout="wide")
    st.title("RF20-VL Prediction Review")

    coco_path = Path(st.text_input("COCO annotation file", "rf20-vl-data/aerial-airport/test/_annotations.coco.json"))
    image_dir = Path(st.text_input("Image directory", "rf20-vl-data/aerial-airport/test"))
    pkl_path = Path(st.text_input("Prediction PKL", "outputs/aerial-airport.pkl"))

    if not coco_path.exists() or not image_dir.exists() or not pkl_path.exists():
        st.info("Enter existing COCO, image directory, and PKL paths.")
        return

    coco = load_json(coco_path)
    preds = load_pkl(pkl_path)
    id_to_name, _ = clean_category_maps(coco)
    pred_by_image = predictions_by_image(preds)
    gt_by_image = annotations_by_image(coco)
    images = sorted(coco.get("images", []), key=lambda image: int(image["id"]))

    table = pd.DataFrame(
        [
            image_summary(image, gt_by_image.get(int(image["id"]), []), pred_by_image.get(int(image["id"]), []))
            for image in images
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    options = [f"{row.image_id} | {row.file_name}" for row in table.itertuples()]
    choice = st.selectbox("Open image", options)
    image_id = int(choice.split("|", 1)[0].strip())
    image = next(item for item in images if int(item["id"]) == image_id)
    image_path = image_dir / str(image["file_name"])
    gt_anns = gt_by_image.get(image_id, [])
    pred_instances = pred_by_image.get(image_id, [])

    cols = st.columns(3)
    for col, title, mode in [(cols[0], "Ground Truth", "gt"), (cols[1], "Prediction", "pred"), (cols[2], "Both", "both")]:
        overlay = render_overlay(image_path, gt_anns, pred_instances, id_to_name, mode)
        col.image(overlay, caption=title, use_container_width=True)
        col.download_button(
            f"Download {title}",
            image_bytes(overlay),
            file_name=f"{image_id}_{mode}.jpg",
            mime="image/jpeg",
            key=f"download_{mode}_{image_id}",
        )


if __name__ == "__main__":
    main()
