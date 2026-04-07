---
title: Annotation QA Env
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---
# 🔍 Annotation QA Environment

An **OpenEnv** environment where a VLM (Vision-Language Model) agent reviews and corrects intentionally-flawed ML annotations on **real COCO val2017 images**. Built for the [Meta OpenEnv × SST Hackathon](https://github.com/meta-pytorch/OpenEnv).

## 🎯 The Challenge

Real-world ML training data is noisy. Annotation teams make mistakes — bounding boxes drift, class labels get swapped, objects get missed. This environment simulates that review pipeline using **500 real images from COCO val2017**:

1. **Agent receives** a real COCO image + current annotations (some are wrong)
2. **Agent visually inspects** the image using a VLM (Qwen2.5-VL-7B-Instruct)
3. **Agent corrects** errors through bbox adjustments, class changes, additions, and removals
4. **Agent submits** and receives a score based on annotation quality improvement

## 📋 Tasks (3 Difficulty Levels)

| Task | Difficulty | Images | Errors | Max Steps |
|------|-----------|--------|--------|-----------| 
| `fix_bboxes` | Easy | 250 | Bbox expansion, shifting, shrinking, spurious, missing | 15 |
| `fix_classes` | Medium | 150 | Bbox errors + class label confusion (car↔truck, dog↔cat) | 20 |
| `batch_audit` | Hard | 100 | Subtle bbox shifts + similar-class confusion + cross-batch | 30 |

## 🏗️ Architecture

```
annotation_qa_env/
├── models.py              ← Action, Observation, State (Pydantic)
├── client.py              ← EnvClient for WebSocket interaction
├── inference.py           ← VLM agent (Qwen2.5-VL-7B via OpenAI API)
├── Dockerfile             ← Container definition
├── server/
│   ├── environment.py     ← Core game logic (reset, step, state)
│   ├── grader.py          ← IoU-based deterministic grading
│   ├── corruption.py      ← Annotation corruption (80 COCO categories)
│   └── app.py             ← FastAPI server
└── data/
    ├── prepare_coco.py    ← One-time COCO preprocessing script
    └── tasks/             ← Pre-processed COCO annotations (~2.5MB)
```

## 🚀 Quick Start

### Install & Run Locally
```bash
cd annotation_qa_env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Inference (VLM)
```bash
export HF_TOKEN="your_hf_token"
python inference.py
```

### Docker
```bash
docker build -t annotation-qa-env:latest .
docker run -d -p 8000:8000 annotation-qa-env:latest
```

## 📊 Grading

The grading function is **deterministic** and returns scores in `[0.0, 1.0]`:

```
Score = (final_quality - initial_quality) / (1.0 - initial_quality)
```

Where `quality` is a weighted composite of:
- **Mean IoU** (40%) — How well do predicted bboxes overlap with gold?
- **Class Accuracy** (30%) — Are class labels correct?
- **Precision** (15%) — Are there spurious annotations?
- **Recall** (15%) — Are there missing annotations?

## 🤖 Actions

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `adjust_bbox` | `annotation_id`, `new_bbox` | Fix a bounding box |
| `change_class` | `annotation_id`, `new_class` | Fix a class label |
| `add_annotation` | `new_bbox`, `new_class` | Add a missing annotation |
| `remove_annotation` | `annotation_id` | Remove a spurious annotation |
| `submit` | (none) | Finalize corrections |

## 📦 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | VLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct` | Vision-Language Model |
| `HF_TOKEN` | — | API key |

## 🖼️ Why Real COCO Images?

This environment uses **500 real images from COCO val2017** with their official annotations:

1. **Real-world complexity**: Actual photographs with occlusion, scale variation, and visual ambiguity
2. **VLM-powered**: The agent can actually *see* the image using Qwen2.5-VL-7B-Instruct
3. **Lightweight**: Only annotations are baked into Docker (~2.5MB); images are fetched from public COCO URLs at inference time
4. **80 COCO categories**: Full diversity of object types
5. **Deterministic grading**: Same seed = same corruptions = reproducible scores

## 📜 License

BSD-3-Clause (matching OpenEnv)
