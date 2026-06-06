# RF20-VL Prompt Bank

These are the prompts used by the training-free RF20-VL submission.

- One prompt per dataset.
- **Class detection behavior**:
  - For most datasets: All classes in the dataset are detected together.
  - **Exceptions**: The following datasets have custom class detection logic and are handled separately:
    - actions
    - soda-bottles
    - all-elements
    - lacrosse-object-detection
    - recode-waste
    - dentalai
    - x-ray-id
- No training, no ICL, no ruler, no refine.
- Output format is Gemini native `box_2d=[ymin,xmin,ymax,xmax]` normalized to `[0,1000]`.
- `scripts/run_gemini_detection.py` appends per-image width/height and converts results to RF20-style PKL.
