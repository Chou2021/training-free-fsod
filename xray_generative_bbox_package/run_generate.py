from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from xray_gen_bbox.common import image_path, load_coco, select_images, write_json
from xray_gen_bbox.gemini_image import call_gemini_image_model, image_mime


def parse_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate colored x-ray detection overlays with Gemini image model.")
    parser.add_argument("--coco", required=True, type=Path, help="Path to _annotations.coco.json")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing images referenced by COCO")
    parser.add_argument("--prompt", required=True, type=Path, help="Prompt text file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output run directory")
    parser.add_argument("--api-base", default=os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com"))
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--first-n", type=int, default=0, help="Only run first N images. 0 means all selected images.")
    parser.add_argument("--image-ids", default="", help="Comma-separated COCO image ids.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    coco = load_coco(args.coco)
    images = select_images(coco, first_n=args.first_n, image_ids=parse_ids(args.image_ids))
    prompt = args.prompt.read_text(encoding="utf-8")
    annotated_dir = args.out_dir / "annotated"
    raw_dir = args.out_dir / "logs" / "raw_api"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    def run_one(image: dict) -> dict:
        image_id = int(image["id"])
        out_path = annotated_dir / f"{image_id}_{Path(str(image['file_name'])).stem}.png"
        raw_path = raw_dir / f"{image_id}.json"
        if out_path.exists() and not args.overwrite:
            return {"image_id": image_id, "ok": True, "skipped": True, "path": str(out_path)}
        img_path = image_path(args.image_dir, image)
        mime = image_mime(img_path)
        seed = args.seed if args.seed >= 0 else None
        images_out, text, raw, error, seconds = call_gemini_image_model(
            args.api_base,
            args.api_key,
            args.model,
            prompt,
            img_path.read_bytes(),
            mime,
            temperature=args.temperature,
            seed=seed,
        )
        write_json(raw_path, {"image_id": image_id, "file_name": image["file_name"], "duration_s": seconds, "error": error, "text": text, "raw_response": raw})
        if error or not images_out:
            return {"image_id": image_id, "ok": False, "error": error or "no generated image"}
        images_out[0].save(out_path)
        return {"image_id": image_id, "ok": True, "path": str(out_path)}

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(run_one, img) for img in images]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="generate"):
            results.append(fut.result())

    write_json(args.out_dir / "logs" / "generate_results.json", sorted(results, key=lambda x: int(x["image_id"])))
    ok = sum(1 for r in results if r.get("ok"))
    print(f"Generated/available: {ok}/{len(results)}")
    print(f"Annotated images: {annotated_dir}")


if __name__ == "__main__":
    main()
