from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


COLOR_TO_LABEL = {
    "red": "DIP",
    "blue": "PIP",
    "yellow": "MCP",
    "green": "Radius",
    "orange": "Ulna",
    "purple": "Wrist",
}

LABEL_TO_CATEGORY_ID = {
    "DIP": 0,
    "MCP": 1,
    "PIP": 2,
    "Radius": 3,
    "Ulna": 4,
    "Wrist": 5,
}


def clamp_bbox(bbox: list[float], width: int, height: int) -> list[float] | None:
    x, y, w, h = [float(v) for v in bbox]
    x = max(0.0, min(float(width), x))
    y = max(0.0, min(float(height), y))
    w = max(0.0, min(float(width) - x, w))
    h = max(0.0, min(float(height) - y, h))
    if w <= 1 or h <= 1:
        return None
    return [x, y, w, h]


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dedupe_boxes(boxes: list[dict[str, Any]], iou_thr: float = 0.35) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for box in sorted(boxes, key=lambda x: float(x["bbox"][2] * x["bbox"][3]), reverse=True):
        duplicate = False
        for old in kept:
            if box["label"] == old["label"] and bbox_iou(box["bbox"], old["bbox"]) > iou_thr:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def contour_boxes(mask: np.ndarray, label: str, min_area: float = 20.0) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w >= 8 and h >= 8 and area >= min_area:
            boxes.append({"label": label, "bbox": [float(x), float(y), float(w), float(h)]})
    return boxes


def yellow_mcp_boxes(mask: np.ndarray) -> list[dict[str, Any]]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    boxes = []
    for idx, contour in enumerate(contours):
        if hierarchy[idx][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w < 8 or h < 8 or area < 20:
            continue
        children = []
        child = hierarchy[idx][2]
        while child != -1:
            cx, cy, cw, ch = cv2.boundingRect(contours[child])
            child_area = cv2.contourArea(contours[child])
            if cw >= 20 and ch >= 20 and child_area >= 200:
                pad = 5
                children.append({
                    "label": "MCP",
                    "bbox": [float(max(0, cx - pad)), float(max(0, cy - pad)), float(cw + 2 * pad), float(ch + 2 * pad)],
                })
            child = hierarchy[child][0]
        if len(children) >= 2:
            boxes.extend(children)
        else:
            boxes.append({"label": "MCP", "bbox": [float(x), float(y), float(w), float(h)]})
    return boxes


def extract_boxes(annotated_image: Image.Image, original_width: int, original_height: int) -> tuple[list[dict[str, Any]], Image.Image]:
    gen_width, gen_height = annotated_image.size
    sx = original_width / max(gen_width, 1)
    sy = original_height / max(gen_height, 1)

    rgb = np.array(annotated_image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    masks = {
        "red": cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 70, 70]), np.array([7, 255, 255])),
            cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255])),
        ),
        "blue": cv2.inRange(hsv, np.array([85, 45, 60]), np.array([130, 255, 255])),
        "yellow": cv2.inRange(hsv, np.array([21, 45, 80]), np.array([45, 255, 255])),
        "green": cv2.inRange(hsv, np.array([42, 55, 50]), np.array([82, 255, 255])),
        "orange": cv2.inRange(hsv, np.array([8, 70, 80]), np.array([20, 255, 255])),
        "purple": cv2.inRange(hsv, np.array([135, 55, 50]), np.array([160, 255, 255])),
    }
    kernel = np.ones((2, 2), np.uint8)
    masks = {k: cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel) for k, v in masks.items()}

    raw_boxes: list[dict[str, Any]] = []
    raw_boxes.extend(contour_boxes(masks["red"], "DIP"))
    raw_boxes.extend(contour_boxes(masks["blue"], "PIP"))
    raw_boxes.extend(yellow_mcp_boxes(masks["yellow"]))
    raw_boxes.extend(contour_boxes(masks["green"], "Radius"))
    raw_boxes.extend(contour_boxes(masks["orange"], "Ulna"))
    raw_boxes.extend(contour_boxes(masks["purple"], "Wrist"))
    raw_boxes = dedupe_boxes(raw_boxes)

    instances = []
    debug = annotated_image.convert("RGB")
    draw = ImageDraw.Draw(debug)
    for box in raw_boxes:
        label = box["label"]
        x, y, w, h = box["bbox"]
        scaled = clamp_bbox([x * sx, y * sy, w * sx, h * sy], original_width, original_height)
        if scaled is None:
            continue
        instances.append({
            "category_id": LABEL_TO_CATEGORY_ID[label],
            "label": label,
            "bbox": scaled,
            "score": 1.0,
        })
        draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=2)
    return instances, debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RF20-style bbox from generated colored x-ray box image.")
    parser.add_argument("--annotated-image", required=True, type=Path)
    parser.add_argument("--original-image", required=True, type=Path, help="Used only for original width/height.")
    parser.add_argument("--image-id", required=True, type=int)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-pkl", type=Path, default=None)
    parser.add_argument("--debug-image", type=Path, default=None)
    args = parser.parse_args()

    original = Image.open(args.original_image).convert("RGB")
    annotated = Image.open(args.annotated_image).convert("RGB")
    instances, debug = extract_boxes(annotated, original.width, original.height)
    instances = [{**inst, "image_id": args.image_id} for inst in instances]
    record = {"image_id": args.image_id, "instances": instances}

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_pkl:
        args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_pkl, "wb") as f:
            pickle.dump([record], f)
    if args.debug_image:
        args.debug_image.parent.mkdir(parents=True, exist_ok=True)
        debug.save(args.debug_image)
    print(f"Extracted boxes: {len(instances)}")


if __name__ == "__main__":
    main()
