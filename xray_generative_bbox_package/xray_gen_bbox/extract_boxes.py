from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def clamp_bbox(bbox: Any, width: int, height: int, min_size: float = 0.0) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in [x, y, w, h]):
        return None
    x = max(0.0, min(float(width), x))
    y = max(0.0, min(float(height), y))
    w = max(0.0, min(float(width) - x, w))
    h = max(0.0, min(float(height) - y, h))
    if w <= min_size or h <= min_size:
        return None
    return [x, y, w, h]


def bbox_iou(a: Any, b: Any) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def merge_duplicate_color_boxes(boxes: list[dict[str, Any]], iou_thr: float = 0.35) -> list[dict[str, Any]]:
    merged = []
    for box in sorted(boxes, key=lambda x: float(x.get("area", x["box"][2] * x["box"][3])), reverse=True):
        duplicate = False
        for kept in merged:
            if box["label"] == kept["label"] and bbox_iou(box["box"], kept["box"]) > iou_thr:
                duplicate = True
                break
        if not duplicate:
            merged.append(box)
    return merged


def extract_color_contour_boxes(mask: np.ndarray, label: str, min_area: float = 20.0) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w >= 8 and h >= 8 and area >= min_area:
            boxes.append({"label": label, "box": [float(x), float(y), float(w), float(h)], "area": float(area), "source": "contour"})
    return boxes


def _runs_1d(values: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(values)
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(a), int(b) + 1) for a, b in zip(starts, ends)]


def extract_rectilinear_color_boxes(mask: np.ndarray, label: str, min_side: int = 10) -> list[dict[str, Any]]:
    h, w = mask.shape[:2]
    binary = mask > 0
    raw_h = []
    for y in range(h):
        for x1, x2 in _runs_1d(binary[y, :]):
            if x2 - x1 >= min_side:
                raw_h.append((x1, float(y), x2))
    raw_v = []
    for x in range(w):
        for y1, y2 in _runs_1d(binary[:, x]):
            if y2 - y1 >= min_side:
                raw_v.append((float(x), y1, y2))

    tol = max(5.0, min(w, h) * 0.015)

    def merge_h(lines: list[tuple[int, float, int]]) -> list[tuple[float, float, float]]:
        groups: list[list[float]] = []
        for x1, y, x2 in sorted(lines, key=lambda v: (v[1], v[0])):
            for g in groups:
                if abs(y - g[1]) <= tol and abs(x1 - g[0]) <= tol and abs(x2 - g[2]) <= tol:
                    n = g[3] + 1
                    g[0] = (g[0] * g[3] + x1) / n
                    g[1] = (g[1] * g[3] + y) / n
                    g[2] = (g[2] * g[3] + x2) / n
                    g[3] = n
                    break
            else:
                groups.append([float(x1), float(y), float(x2), 1.0])
        return [(g[0], g[1], g[2]) for g in groups]

    def merge_v(lines: list[tuple[float, int, int]]) -> list[tuple[float, float, float]]:
        groups: list[list[float]] = []
        for x, y1, y2 in sorted(lines, key=lambda v: (v[0], v[1])):
            for g in groups:
                if abs(x - g[0]) <= tol and abs(y1 - g[1]) <= tol and abs(y2 - g[2]) <= tol:
                    n = g[3] + 1
                    g[0] = (g[0] * g[3] + x) / n
                    g[1] = (g[1] * g[3] + y1) / n
                    g[2] = (g[2] * g[3] + y2) / n
                    g[3] = n
                    break
            else:
                groups.append([float(x), float(y1), float(y2), 1.0])
        return [(g[0], g[1], g[2]) for g in groups]

    h_lines = merge_h(raw_h)
    v_lines = merge_v(raw_v)
    candidates = []
    for i, top in enumerate(h_lines):
        tx1, ty, tx2 = top
        for bottom in h_lines[i + 1:]:
            bx1, by, bx2 = bottom
            if by <= ty + min_side:
                continue
            if abs(tx1 - bx1) > tol or abs(tx2 - bx2) > tol:
                continue
            x1 = (tx1 + bx1) / 2.0
            x2 = (tx2 + bx2) / 2.0
            left = [v for v in v_lines if abs(v[0] - x1) <= tol and v[1] <= ty + tol and v[2] >= by - tol]
            right = [v for v in v_lines if abs(v[0] - x2) <= tol and v[1] <= ty + tol and v[2] >= by - tol]
            if left and right and x2 - x1 >= min_side:
                candidates.append({"label": label, "box": [x1, ty, x2 - x1, by - ty], "area": (x2 - x1) * (by - ty), "source": "rectilinear"})
    return merge_duplicate_color_boxes(candidates, iou_thr=0.75)


def extract_yellow_boxes_with_inner_split(mask: np.ndarray) -> list[dict[str, Any]]:
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
                children.append({"label": "MCP", "box": [float(max(0, cx - pad)), float(max(0, cy - pad)), float(cw + 2 * pad), float(ch + 2 * pad)], "area": float(child_area), "source": "inner_split"})
            child = hierarchy[child][0]
        boxes.extend(children if len(children) >= 2 else [{"label": "MCP", "box": [float(x), float(y), float(w), float(h)], "area": float(area), "source": "external"}])
    return merge_duplicate_color_boxes(boxes, iou_thr=0.55)


def extract_generated_color_boxes(
    annotated_img: Image.Image,
    original_width: int,
    original_height: int,
    color_to_label: dict[str, str],
    name_to_id: dict[str, int],
    split_yellow: bool = True,
    extraction_mode: str = "contour",
) -> tuple[list[dict[str, Any]], Image.Image, dict[str, Any]]:
    gen_width, gen_height = annotated_img.size
    sx = original_width / max(gen_width, 1)
    sy = original_height / max(gen_height, 1)

    rgb = np.array(annotated_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    masks = {
        "red": cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 70, 70]), np.array([7, 255, 255])), cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255]))),
        "blue": cv2.inRange(hsv, np.array([85, 45, 60]), np.array([130, 255, 255])),
        "yellow": cv2.inRange(hsv, np.array([21, 45, 80]), np.array([45, 255, 255])),
        "green": cv2.inRange(hsv, np.array([42, 55, 50]), np.array([82, 255, 255])),
        "orange": cv2.inRange(hsv, np.array([8, 70, 80]), np.array([20, 255, 255])),
        "purple": cv2.inRange(hsv, np.array([135, 55, 50]), np.array([160, 255, 255])),
    }
    kernel = np.ones((2, 2), np.uint8)
    masks = {k: cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel) for k, v in masks.items()}

    boxes: list[dict[str, Any]] = []
    for color, label in color_to_label.items():
        if not label or color not in masks:
            continue
        if color == "yellow" and split_yellow and label == "MCP":
            boxes.extend(extract_yellow_boxes_with_inner_split(masks[color]))
        elif color in {"green", "orange", "purple"} and extraction_mode == "rectilinear":
            boxes.extend(extract_rectilinear_color_boxes(masks[color], label))
        else:
            boxes.extend(extract_color_contour_boxes(masks[color], label))
    boxes = merge_duplicate_color_boxes(boxes)

    instances: list[dict[str, Any]] = []
    debug = annotated_img.convert("RGB")
    draw = ImageDraw.Draw(debug)
    draw_colors = {
        "DIP": (255, 0, 0),
        "PIP": (0, 128, 255),
        "MCP": (255, 200, 0),
        "Radius": (0, 200, 0),
        "Ulna": (255, 140, 0),
        "Wrist": (160, 0, 200),
    }
    for box in boxes:
        x, y, w, h = box["box"]
        label = str(box["label"])
        bbox = clamp_bbox([x * sx, y * sy, w * sx, h * sy], original_width, original_height, min_size=1.0)
        cat_id = name_to_id.get(label)
        if bbox is None or cat_id is None:
            continue
        instances.append({"category_id": int(cat_id), "label": label, "bbox": bbox, "score": 1.0, "source": box.get("source", "")})
        gx, gy, gw, gh = [int(round(v)) for v in box["box"]]
        draw.rectangle([gx, gy, gx + gw, gy + gh], outline=draw_colors.get(label, (255, 255, 255)), width=3)

    meta = {"generated_size": [gen_width, gen_height], "original_size": [original_width, original_height], "scale": [sx, sy], "raw_box_count": len(boxes), "extraction_mode": extraction_mode}
    return instances, debug, meta
