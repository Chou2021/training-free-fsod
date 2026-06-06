from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import json

from xray_gen_bbox.common import COLOR_PRESETS, category_maps, image_path, load_coco, save_pkl, select_images, write_json
from xray_gen_bbox.extract_boxes import extract_generated_color_boxes


def parse_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def load_color_map(value: str, map_file: Path | None) -> dict[str, str]:
    if map_file is not None:
        payload = json.loads(map_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("--color-map-file must contain a JSON object")
        return {str(k): str(v) for k, v in payload.items()}
    return COLOR_PRESETS[value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract COCO bbox pkl from generated colored x-ray overlays.")
    parser.add_argument("--coco", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, help="Directory produced by run_generate.py")
    parser.add_argument("--out-pkl", required=True, type=Path)
    parser.add_argument("--preset", choices=sorted(COLOR_PRESETS), default="all6")
    parser.add_argument(
        "--color-map-file",
        type=Path,
        default=None,
        help="Optional JSON object mapping color names to COCO category names. Overrides --preset.",
    )
    parser.add_argument("--extraction-mode", choices=["contour", "rectilinear"], default="contour")
    parser.add_argument("--no-split-yellow", action="store_true")
    parser.add_argument("--first-n", type=int, default=0)
    parser.add_argument("--image-ids", default="")
    args = parser.parse_args()

    coco = load_coco(args.coco)
    _, name_to_id = category_maps(coco)
    images = select_images(coco, first_n=args.first_n, image_ids=parse_ids(args.image_ids))
    color_to_label = load_color_map(args.preset, args.color_map_file)
    annotated_dir = args.run_dir / "annotated"
    extracted_dir = args.run_dir / "extracted"
    debug_dir = extracted_dir / "debug"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    records = []
    failures = []
    for image in tqdm(images, desc="extract"):
        image_id = int(image["id"])
        ann_path = annotated_dir / f"{image_id}_{Path(str(image['file_name'])).stem}.png"
        if not ann_path.exists():
            failures.append({"image_id": image_id, "file_name": image["file_name"], "error": "missing annotated image"})
            continue
        try:
            # Open original only to verify path and use COCO dimensions for coordinate scaling.
            image_path(args.image_dir, image)
            annotated = Image.open(ann_path).convert("RGB")
            instances, debug_img, meta = extract_generated_color_boxes(
                annotated,
                int(image["width"]),
                int(image["height"]),
                color_to_label,
                name_to_id,
                split_yellow=not args.no_split_yellow,
                extraction_mode=args.extraction_mode,
            )
            fixed = [{**inst, "image_id": image_id} for inst in instances]
            records.append({"image_id": image_id, "instances": fixed})
            debug_img.save(debug_dir / f"{image_id}_{Path(str(image['file_name'])).stem}.jpg", quality=92)
            write_json(extracted_dir / f"{image_id}.json", {"image_id": image_id, "file_name": image["file_name"], "instances": fixed, "meta": meta, "color_to_label": color_to_label})
        except Exception as exc:
            failures.append({"image_id": image_id, "file_name": image["file_name"], "error": str(exc)})

    save_pkl(args.out_pkl, records)
    write_json(args.run_dir / "logs" / "extract_failures.json", failures)
    print(f"Wrote pkl: {args.out_pkl}")
    print(f"Images: {len(records)}, boxes: {sum(len(r.get('instances', []) or []) for r in records)}, failures: {len(failures)}")


if __name__ == "__main__":
    main()
