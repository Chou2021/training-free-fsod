# X-ray 识别最小代码包

只保留三部分：

```text
xray_bbox_recognition_minimal/
├── prompts/
│   └── xray_colored_box_prompt.txt
├── call_model.py
├── extract_boxes.py
└── README_zh.md
```

## 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 2. 设置 API key

代码不会保存 key。运行前设置环境变量：

```bash
export GEMINI_API_KEY='你的key'
export GEMINI_API_BASE='https://generativelanguage.googleapis.com'
export GEMINI_IMAGE_MODEL='gemini-3.1-flash-image-preview'
```

## 3. 调用生成模型画彩色框

```bash
python call_model.py \
  --image /path/to/xray.jpg \
  --prompt prompts/xray_colored_box_prompt.txt \
  --out-image outputs/xray_annotated.png \
  --raw-json outputs/raw_response.json
```

输出：

```text
outputs/xray_annotated.png
```

这张图是在原始 X 光图上画了彩色检测框。

## 4. 从彩色框图提取 bbox

```bash
python extract_boxes.py \
  --annotated-image outputs/xray_annotated.png \
  --original-image /path/to/xray.jpg \
  --image-id 0 \
  --out-json outputs/pred.json \
  --out-pkl outputs/pred.pkl \
  --debug-image outputs/debug_extract.png
```

输出 bbox 是 COCO 格式：

```text
[x, y, width, height]
```

输出 pkl 格式：

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

这个示例是单张图的 pkl。完整数据集需要把每张图都跑一遍，然后把所有 image record 合并成一个 list：

```python
[
  {"image_id": 0, "instances": [...]},
  {"image_id": 1, "instances": [...]},
  {"image_id": 2, "instances": [...]},
]
```

真实 API 单图测试示例：

```text
image_id: 0
生成图尺寸: 944x1116
原图尺寸: 433x512
提取框数量: 17
类别数量: DIP 4, PIP 5, MCP 5, Radius 1, Ulna 1, Wrist 1
```

## 5. 类别和颜色

| 类别 | category_id | 颜色 |
|---|---:|---|
| DIP | 0 | red |
| MCP | 1 | yellow |
| PIP | 2 | blue |
| Radius | 3 | green |
| Ulna | 4 | orange |
| Wrist | 5 | purple |

## 6. 注意

- 这个最小包只处理单张图片。
- 批量跑可以外层自己写循环。
- `extract_boxes.py` 会自动把生成图坐标缩放回原图尺寸。
- `score` 默认是 `1.0`。
- 如果官方提交类别 id 不同，需要提交前再做 category_id 映射。
