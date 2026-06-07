# X-ray colored-box extraction

This folder contains the minimal two-step pipeline used for hand X-ray boxes in `x-ray-id`:

1. Use a Gemini image model to draw colored box outlines on one X-ray image.
2. Use OpenCV to extract those colored outlines as COCO-style boxes.

The scripts do not contain an API key. Set `GEMINI_API_KEY` in the shell.

## Files

```text
generative_bbox_package/
├── prompts/
│   └── xray_colored_box_prompt.txt
├── call_model.py
├── extract_boxes.py
├── README.md
└── README_zh.md
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Environment

```bash
export GEMINI_API_KEY='your_key_here'
export GEMINI_API_BASE='https://generativelanguage.googleapis.com'
export GEMINI_IMAGE_MODEL='gemini-3.1-flash-image-preview'
```

`GEMINI_API_BASE` and `GEMINI_IMAGE_MODEL` match the defaults used by `call_model.py`, so only `GEMINI_API_KEY` is strictly required when using the official endpoint.

## 1. Generate the annotated image

```bash
python call_model.py \
  --image /path/to/xray.jpg \
  --prompt prompts/xray_colored_box_prompt.txt \
  --out-image outputs/xray_annotated.png \
  --raw-json outputs/raw_response.json
```

The output should be the original X-ray with colored rectangle outlines drawn on top.

## 2. Extract boxes

```bash
python extract_boxes.py \
  --annotated-image outputs/xray_annotated.png \
  --original-image /path/to/xray.jpg \
  --image-id 0 \
  --out-json outputs/pred.json \
  --out-pkl outputs/pred.pkl \
  --debug-image outputs/debug_extract.png
```

`--original-image` is used for the original width and height. If the generated image is resized by the model, the extracted boxes are scaled back to the original coordinate system.

Boxes are saved as COCO `xywh`:

```text
[x, y, width, height]
```

The pickle contains a list of image records:

```python
[
  {
    "image_id": 0,
    "instances": [
      {
        "image_id": 0,
        "category_id": 0,
        "label": "DIP",
        "bbox": [x, y, width, height],
        "score": 1.0
      }
    ]
  }
]
```

That example is a one-image pkl. For a full dataset, run this script for each image and combine all image records into one list:

```python
[
  {"image_id": 0, "instances": [...]},
  {"image_id": 1, "instances": [...]},
  {"image_id": 2, "instances": [...]},
]
```

Example from one real API test:

```text
image_id: 0
generated image size: 944x1116
original image size: 433x512
extracted boxes: 17
class counts: DIP 4, PIP 5, MCP 5, Radius 1, Ulna 1, Wrist 1
```

## Labels

| label | category_id | color |
|---|---:|---|
| DIP | 0 | red |
| MCP | 1 | yellow |
| PIP | 2 | blue |
| Radius | 3 | green |
| Ulna | 4 | orange |
| Wrist | 5 | purple |

## Notes

- This is single-image code. For a dataset, call the two scripts in a loop.
- Scores are fixed at `1.0`.
- The extractor expects the generated image to use the colors in `prompts/xray_colored_box_prompt.txt`.
- Check category IDs before submission. Some COCO exports include a dummy class that shifts labels by one.
