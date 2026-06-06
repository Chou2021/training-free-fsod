# Generative Colored-box Detection

This package contains the command-line workflow used for `x-ray-id` and adapted for `dentalai`.

The workflow is:

1. Ask a Gemini image generation model to draw colored rectangle outlines on the input image.
2. Use OpenCV color thresholding to extract those rectangles.
3. Convert rectangles to RF20-style PKL predictions.

No Streamlit code or API keys are required.

## Setup

From this package directory:

```bash
python -m pip install -r requirements.txt
export GEMINI_API_KEY="YOUR_KEY"
```

Optional endpoint/model overrides:

```bash
export GEMINI_API_BASE="https://generativelanguage.googleapis.com"
export GEMINI_IMAGE_MODEL="gemini-3.1-flash-image-preview"
```

If your API provider exposes a Gemini-compatible image model under another name, pass `--model` explicitly.

## X-ray-id Example

Generate colored overlays for the first 10 images:

```bash
python run_generate.py \
  --coco ../rf20-vl-data/x-ray-id/test/_annotations.coco.json \
  --image-dir ../rf20-vl-data/x-ray-id/test \
  --prompt prompts/xray_overlay_prompt.txt \
  --out-dir outputs/xray-id-first10 \
  --first-n 10 \
  --workers 4 \
  --temperature 0
```

Extract all six classes:

```bash
python run_extract.py \
  --coco ../rf20-vl-data/x-ray-id/test/_annotations.coco.json \
  --image-dir ../rf20-vl-data/x-ray-id/test \
  --run-dir outputs/xray-id-first10 \
  --out-pkl outputs/xray-id-first10/pkls/xray-id.pkl \
  --preset all6 \
  --first-n 10
```

For wrist/forearm boxes, `--extraction-mode rectilinear` can be more stable:

```bash
python run_extract.py \
  --coco ../rf20-vl-data/x-ray-id/test/_annotations.coco.json \
  --image-dir ../rf20-vl-data/x-ray-id/test \
  --run-dir outputs/xray-id-first10 \
  --out-pkl outputs/xray-id-first10/pkls/xray-wrist-forearm.pkl \
  --preset wrist_forearm \
  --extraction-mode rectilinear \
  --first-n 10
```

## DentalAI Example

```bash
python run_generate.py \
  --coco ../rf20-vl-data/dentalai/test/_annotations.coco.json \
  --image-dir ../rf20-vl-data/dentalai/test \
  --prompt prompts/dental_overlay_prompt.txt \
  --out-dir outputs/dentalai-first10 \
  --first-n 10 \
  --workers 4 \
  --temperature 0

python run_extract.py \
  --coco ../rf20-vl-data/dentalai/test/_annotations.coco.json \
  --image-dir ../rf20-vl-data/dentalai/test \
  --run-dir outputs/dentalai-first10 \
  --out-pkl outputs/dentalai-first10/pkls/dentalai.pkl \
  --preset dentalai \
  --first-n 10
```

## Merge Multiple PKLs

```bash
python merge_pkls.py \
  --inputs outputs/xray-joints.pkl outputs/xray-wrist-forearm.pkl \
  --out-pkl outputs/xray-merged.pkl \
  --dedupe-iou 0.8
```

## Color Presets

```text
all6: DIP, PIP, MCP, Radius, Ulna, Wrist
joints: DIP, PIP, MCP
wrist_forearm: Radius, Ulna, Wrist
forearm_bones: Radius, Ulna
dentalai: Cavity, Fillings, Impacted Tooth, Implant
```

The extractor also accepts a custom JSON color map:

```bash
python run_extract.py ... --color-map-file my_colors.json
```

`my_colors.json` should look like:

```json
{"red": "Class A", "blue": "Class B"}
```

Supported color names are `red`, `blue`, `yellow`, `green`, `orange`, and `purple`.
