# Training-free FSOD

This repository contains the code and prompts used for our CVPR 2026 RF20-VL few-shot object detection submission. The method is training-free: it uses vision-language model prompting, lightweight JSON-to-PKL conversion, and a small amount of deterministic post-processing.

The repository is intentionally small. It is meant to let organizers reproduce representative classes or a subset of images without needing our full Streamlit experiment workspace.

## Repository Layout

```text
.
├── prompt_bank/                         # Dataset prompts
├── scripts/
│   ├── run_gemini_detection.py          # Main VLM detection runner
│   ├── merge_category_predictions.py    # Merge single-class outputs into a base PKL
│   ├── weighted_box_fusion.py           # Fuse loose/tight boxes, used for soda-bottles
│   └── rescore_with_gpt_confidence.py   # Optional GPT confidence rescoring
├── generative_bbox_package/             # Colored-box generation and extraction
├── configs/dataset_strategies.json      # Machine-readable strategy summary
├── streamlit_review_app.py              # Optional lightweight visual review UI
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

By default, scripts use the official Gemini API base `https://generativelanguage.googleapis.com`. If you use an OpenAI-compatible or Gemini-compatible gateway, override:

```bash
export GEMINI_API_BASE="https://your-gateway.example.com"
export GEMINI_MODEL="gemini-3.5-flash"
```

## Input and Output Format

The runner expects a Roboflow/RF20 COCO split:

```text
rf20-vl-data/<dataset>/test/
├── _annotations.coco.json
├── image_1.jpg
└── ...
```

The model is prompted to return Gemini-native boxes:

```json
[{"label": "class name", "box_2d": [ymin, xmin, ymax, xmax], "confidence": 1}]
```

`scripts/run_gemini_detection.py` converts this to the RF20-style PKL:

```python
[
  {
    "image_id": 0,
    "instances": [
      {
        "image_id": 0,
        "category_id": 1,
        "bbox": [x, y, width, height],
        "score": 1.0
      }
    ]
  }
]
```

## Strategy Summary

Most datasets are run as single-pass multi-class detection using one prompt per dataset:

```text
aerial-airport, aquarium-combined, defect-detection, flir-camera-objects,
gwhd2021, new-defects-in-wood, orionproducts, paper-parts,
the-dreidel-project, trail-camera, water-meter, wb-prova, wildfire-smoke
```

Special cases:

```text
actions, lacrosse-object-detection, recode-waste:
  combine a multi-class pass with one or more single-class passes.

soda-bottles:
  run loose and tight prompts, then fuse matched boxes with weighted averaging.

all-elements:
  has a multi-class prompt plus per-class prompts in prompt_bank/all-elements/.

x-ray-id, dentalai:
  use the generative colored-box workflow in generative_bbox_package/.
```

See `configs/dataset_strategies.json` for a machine-readable index of the prompts used by each strategy.

## Reproduce a Standard Single-pass Dataset

Example for the first 10 test images of `aerial-airport`:

```bash
python scripts/run_gemini_detection.py \
  --coco rf20-vl-data/aerial-airport/test/_annotations.coco.json \
  --image-dir rf20-vl-data/aerial-airport/test \
  --prompt prompt_bank/aerial-airport/aerial-airport.txt \
  --output outputs/aerial-airport-first10.pkl \
  --dataset aerial-airport \
  --first-n 10 \
  --workers 4
```

Run selected image ids:

```bash
python scripts/run_gemini_detection.py \
  --coco rf20-vl-data/gwhd2021/test/_annotations.coco.json \
  --image-dir rf20-vl-data/gwhd2021/test \
  --prompt prompt_bank/gwhd2021/gwhd2021.txt \
  --output outputs/gwhd2021-selected.pkl \
  --dataset gwhd2021 \
  --image-ids 0,1,2
```

## Combined Multi-class and Single-class Datasets

The pattern is:

1. Run the base multi-class prompt.
2. Run a single-class prompt with `--category`.
3. Replace or append that category using `merge_category_predictions.py`.

Example for `recode-waste` aggregate:

```bash
python scripts/run_gemini_detection.py \
  --coco rf20-vl-data/recode-waste/test/_annotations.coco.json \
  --image-dir rf20-vl-data/recode-waste/test \
  --prompt prompt_bank/recode-waste/recode-waste.txt \
  --output outputs/recode-base.pkl \
  --dataset recode-waste \
  --first-n 10

python scripts/run_gemini_detection.py \
  --coco rf20-vl-data/recode-waste/test/_annotations.coco.json \
  --image-dir rf20-vl-data/recode-waste/test \
  --prompt prompt_bank/recode-waste/recode-waste-aggregate.txt \
  --output outputs/recode-aggregate.pkl \
  --dataset recode-waste \
  --category aggregate \
  --first-n 10

python scripts/merge_category_predictions.py \
  --base outputs/recode-base.pkl \
  --source outputs/recode-aggregate.pkl \
  --output outputs/recode-merged.pkl \
  --coco rf20-vl-data/recode-waste/test/_annotations.coco.json \
  --category aggregate
```

The same command pattern applies to:

```text
actions: ball, Attack, Block, Defense, Serve, Set
lacrosse-object-detection: Longpole
recode-waste: aggregate, metal
all-elements: Button, Check box, Checked Radio button, Checked box,
              Dropdown box, Dropdown expand, Icon, Radio button, Scroll bar, Text box
```

Use `--mode append` instead of `replace` when you want to add a single-class pass without removing the base class predictions.

## Soda-bottles Weighted Box Fusion

Run loose and tight prompts, then fuse:

```bash
python scripts/run_gemini_detection.py \
  --coco rf20-vl-data/soda-bottles/test/_annotations.coco.json \
  --image-dir rf20-vl-data/soda-bottles/test \
  --prompt prompt_bank/soda-bottles/soda-bottles-loose.txt \
  --output outputs/soda-loose.pkl \
  --dataset soda-bottles \
  --first-n 10

python scripts/run_gemini_detection.py \
  --coco rf20-vl-data/soda-bottles/test/_annotations.coco.json \
  --image-dir rf20-vl-data/soda-bottles/test \
  --prompt prompt_bank/soda-bottles/soda-bottles-tight.txt \
  --output outputs/soda-tight.pkl \
  --dataset soda-bottles \
  --first-n 10

python scripts/weighted_box_fusion.py \
  --first outputs/soda-loose.pkl \
  --second outputs/soda-tight.pkl \
  --output outputs/soda-fused.pkl \
  --weight 0.5 \
  --keep-unmatched first
```

## GPT Confidence Rescoring

The technical report also describes a confidence calibration step. We used a GPT vision model to judge each candidate box and replace the original score with a value in `[0, 1]`.

Set an API key and model:

```bash
export GPT_API_KEY="YOUR_KEY"
export GPT_MODEL="gpt-5.4"
```

Then rescore an existing PKL:

```bash
python scripts/rescore_with_gpt_confidence.py \
  --coco rf20-vl-data/aerial-airport/test/_annotations.coco.json \
  --image-dir rf20-vl-data/aerial-airport/test \
  --input-pkl outputs/aerial-airport-first10.pkl \
  --output-pkl outputs/aerial-airport-first10-gpt-scores.pkl \
  --dataset aerial-airport \
  --max-boxes 20 \
  --raw-dir raw_outputs/aerial-airport-gpt-scores
```

For datasets where one-shot visual reference helps, add a green-box train exemplar:

```bash
python scripts/rescore_with_gpt_confidence.py \
  --coco rf20-vl-data/recode-waste/test/_annotations.coco.json \
  --image-dir rf20-vl-data/recode-waste/test \
  --input-pkl outputs/recode-merged.pkl \
  --output-pkl outputs/recode-merged-gpt-scores.pkl \
  --dataset recode-waste \
  --use-train-exemplar
```

The script keeps `image_id`, `category_id`, and `bbox` unchanged; only `score` is updated. It can resume from an existing output PKL.

## X-ray-id and DentalAI

For these datasets we used a generative colored-box workflow: the image model draws colored rectangle outlines on the image, and OpenCV extracts those colored boxes into COCO coordinates.

See `generative_bbox_package/README.md` for commands.

## Visual Review

The optional Streamlit viewer compares a PKL against local COCO annotations:

```bash
streamlit run streamlit_review_app.py
```

It does not call any API. It only loads local image, COCO, and PKL files for inspection.

## Notes

- The scripts are designed for reproducibility on a subset of images/classes. Running all datasets requires API access and the RF20-VL image files.
- No API keys or generated prediction PKLs are included.
- Add your preferred license before publishing the repository publicly.
