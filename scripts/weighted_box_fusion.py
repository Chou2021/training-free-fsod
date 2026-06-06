#!/usr/bin/env python3
"""Weighted merge for two RF20-style PKL files.

This was used for soda-bottles: one prompt produced loose bottle boxes and a
second prompt produced tight cap/neck boxes. Matched boxes are fused by:

    fused_bbox = (1 - weight) * first_bbox + weight * second_bbox
"""

from __future__ import annotations

import argparse
import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Match:
    first_index: int
    second_index: int
    iou: float


def load_pkl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a list, got {type(data).__name__}")
    return data


def save_pkl(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def bbox4(instance: dict[str, Any], context: str) -> list[float]:
    bbox = instance.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        raise ValueError(f"{context} has invalid bbox: {bbox!r}")
    return [float(value) for value in bbox[:4]]


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def index_by_image(data: list[dict[str, Any]], label: str) -> dict[Any, dict[str, Any]]:
    indexed: dict[Any, dict[str, Any]] = {}
    for row in data:
        image_id = row.get("image_id")
        if image_id in indexed:
            raise ValueError(f"{label} has duplicate image_id={image_id}")
        indexed[image_id] = row
    return indexed


def match_by_iou(
    first_instances: list[dict[str, Any]],
    second_instances: list[dict[str, Any]],
    *,
    same_category: bool,
    iou_threshold: float,
) -> list[Match]:
    candidates: list[tuple[float, int, int]] = []
    for first_index, first_inst in enumerate(first_instances):
        first_box = bbox4(first_inst, f"first index={first_index}")
        for second_index, second_inst in enumerate(second_instances):
            if same_category and first_inst.get("category_id") != second_inst.get("category_id"):
                continue
            second_box = bbox4(second_inst, f"second index={second_index}")
            iou = bbox_iou(first_box, second_box)
            if iou >= iou_threshold:
                candidates.append((iou, first_index, second_index))
    candidates.sort(reverse=True)

    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: list[Match] = []
    for iou, first_index, second_index in candidates:
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches.append(Match(first_index, second_index, iou))
    return sorted(matches, key=lambda match: match.first_index)


def weighted_bbox(first_bbox: list[float], second_bbox: list[float], second_weight: float) -> list[float]:
    first_weight = 1.0 - second_weight
    return [first_weight * a + second_weight * b for a, b in zip(first_bbox, second_bbox)]


def combine_score(first_inst: dict[str, Any], second_inst: dict[str, Any], weight: float, policy: str) -> float:
    first_score = float(first_inst.get("score", 1.0))
    second_score = float(second_inst.get("score", 1.0))
    if policy == "first":
        return first_score
    if policy == "second":
        return second_score
    if policy == "max":
        return max(first_score, second_score)
    if policy == "average":
        return (1.0 - weight) * first_score + weight * second_score
    raise ValueError(f"Unknown score policy: {policy}")


def fuse_records(
    first_data: list[dict[str, Any]],
    second_data: list[dict[str, Any]],
    *,
    weight: float,
    iou_threshold: float,
    same_category: bool,
    keep_unmatched: str,
    score_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    second_by_image = index_by_image(second_data, "second")
    output = copy.deepcopy(first_data)
    stats: dict[str, float | int] = {
        "images": len(first_data),
        "first_instances": 0,
        "second_instances": 0,
        "matched": 0,
        "unmatched_first": 0,
        "unmatched_second": 0,
        "output_instances": 0,
        "mean_match_iou": 0.0,
    }
    iou_sum = 0.0

    for row_index, first_row in enumerate(first_data):
        image_id = first_row.get("image_id")
        if image_id not in second_by_image:
            raise ValueError(f"image_id={image_id} is missing from second PKL")
        first_instances = first_row.get("instances", []) or []
        second_instances = second_by_image[image_id].get("instances", []) or []
        matches = match_by_iou(
            first_instances,
            second_instances,
            same_category=same_category,
            iou_threshold=iou_threshold,
        )
        used_first = {match.first_index for match in matches}
        used_second = {match.second_index for match in matches}
        merged_instances: list[dict[str, Any]] = []

        for match in matches:
            first_inst = first_instances[match.first_index]
            second_inst = second_instances[match.second_index]
            merged = copy.deepcopy(first_inst)
            merged["image_id"] = image_id
            merged["bbox"] = weighted_bbox(
                bbox4(first_inst, f"first image_id={image_id}, index={match.first_index}"),
                bbox4(second_inst, f"second image_id={image_id}, index={match.second_index}"),
                weight,
            )
            merged["score"] = combine_score(first_inst, second_inst, weight, score_policy)
            merged_instances.append(merged)

        if keep_unmatched in {"first", "both"}:
            merged_instances.extend(copy.deepcopy(inst) for idx, inst in enumerate(first_instances) if idx not in used_first)
        if keep_unmatched in {"second", "both"}:
            merged_instances.extend(copy.deepcopy(inst) for idx, inst in enumerate(second_instances) if idx not in used_second)

        output[row_index]["instances"] = merged_instances
        stats["first_instances"] += len(first_instances)
        stats["second_instances"] += len(second_instances)
        stats["matched"] += len(matches)
        stats["unmatched_first"] += len(first_instances) - len(used_first)
        stats["unmatched_second"] += len(second_instances) - len(used_second)
        stats["output_instances"] += len(merged_instances)
        iou_sum += sum(match.iou for match in matches)

    if stats["matched"]:
        stats["mean_match_iou"] = iou_sum / float(stats["matched"])
    return output, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weighted bbox fusion for two RF20-style PKLs.")
    parser.add_argument("--first", required=True, type=Path, help="First PKL, e.g. soda-bottles loose prompt output.")
    parser.add_argument("--second", required=True, type=Path, help="Second PKL, e.g. soda-bottles tight prompt output.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weight", type=float, default=0.5, help="Second bbox weight.")
    parser.add_argument("--iou-threshold", type=float, default=0.05)
    parser.add_argument("--allow-category-mismatch", action="store_true")
    parser.add_argument("--keep-unmatched", choices=["first", "second", "both", "none"], default="first")
    parser.add_argument("--score-policy", choices=["first", "second", "average", "max"], default="second")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.weight <= 1.0:
        raise ValueError("--weight must be between 0 and 1")
    fused, stats = fuse_records(
        load_pkl(args.first),
        load_pkl(args.second),
        weight=args.weight,
        iou_threshold=args.iou_threshold,
        same_category=not args.allow_category_mismatch,
        keep_unmatched=args.keep_unmatched,
        score_policy=args.score_policy,
    )
    print(f"First: {args.first}")
    print(f"Second: {args.second}")
    print(f"Output: {args.output}")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    if not args.dry_run:
        save_pkl(args.output, fused)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
