# 📘 Knowledge Distillation for Lightweight 2D Object Detection**

## 🚀 Project Overview

This repository implements a modular pipeline to study **Knowledge Distillation (KD)** for lightweight 2D object detection using the Pascal VOC dataset.

We investigate how feature-level, logit-level, and combined distillation can improve lightweight detectors using:

**Teacher Models**

* **DINO-V2** (feature distillation)
* **CLIP ViT** (logit distillation)

**Student Models**

* **ResNet-18** backbone + FasterRCNN
* **MobileNetV3-Small** backbone + FasterRCNN
* **ViT-Tiny** backbone + FasterRCNN

The system supports:

* Swappable student backbones
* Pluggable teacher models
* Toggleable distillation components
* Integrated VOC mAP evaluation (0.50–0.95)
* TensorBoard logging
* Visualizations (predictions + DINO heatmaps)

---

# 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/CSC2503_Project.git
cd CSC2503_Project
```

### 2. Create your conda environment

```bash
conda create -n csc2503 python=3.11 -y
conda activate csc2503
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pascal VOC data

Place Pascal VOC inside:

```
project_root/data/VOCdevkit/
```

Download this: https://drive.google.com/file/d/1UCy6hJulnhEoxZbZvtBNUnb6mzLYy0Yb/view?usp=sharing

And unzip it inside the `data` folder.

In the end you should have `VOCdevkit` (extracted) under `data`

Dataset structure should look like:

```
data/
└── VOCdevkit/
    ├── VOC2007/
    └── VOC2012/
```

---

# 🚂 Datasets

We use a custom dataset wrapper:

```
src/dataset.py
```

It:

* Loads VOC 2007 / 2012
* Produces **student_img**, **teacher_img**, **target_dict**
* Resizes both images to **480×480**
* Applies correct bounding box scaling
* Collates batches into `(B, 3, 480, 480)` tensors

---

# 🧩 Training

Training is controlled via:

```
src/train.py
```

All models, KD flags, and hyperparameters are CLI-controlled.

---

## 🏁 **1. Baseline training (no distillation)**

### ResNet-18 baseline

```bash
python train.py --backbone resnet18 --epochs 20 --batch-size 8
```

Include flag `--eval-every-epoch` to calculate mAP each epoch, or `--eval-final-only`
to only calculate mAP at the end of training.

### MobileNetV3-Small baseline

```bash
python train.py --backbone mobilenet_v3_small --epochs 20
```

### ViT-Tiny baseline

```bash
python train.py --backbone vit_tiny --epochs 20
```

Logs saved under:

```
runs/<run_name>/
```

Checkpoints under:

```
checkpoints/
```

---

## 🔥 **2. Enable knowledge distillation**

### Feature KD only (DINO)

```bash
python train.py --backbone resnet18 --distill --beta 1 --gamma 0
```

### Logit KD only (CLIP)

```bash
python train.py --backbone resnet18 --distill --beta 0 --gamma 1
```

### Full KD (feature + logit)

```bash
python train.py --backbone resnet18 --distill --beta 1 --gamma 1
```

Parameters:

| Flag        | Meaning                             |
| ----------- | ----------------------------------- |
| `--alpha`   | Detection loss weight (default 1.0) |
| `--beta`    | Feature KD weight                   |
| `--gamma`   | Logit KD weight                     |
| `--distill` | Enables DINO + CLIP teachers        |

---

## 🧪 Evaluation (VOC mAP)

You can evaluate any checkpoint using:

```bash
python eval.py \
  --checkpoint checkpoints/student_latest.pth \
  --backbone resnet18
```

This prints:

* mAP@0.50
* mAP@0.55
* … up to mAP@0.95
* Overall mAP@[0.50:0.95]

Evaluation is also automatically run:

* per epoch (`--eval-every-epoch`)
* or once at the end (`--eval-final-only`)

---

# 🧱 Architecture

```
src/
├── dataset.py             # VOC dataset + resizing + scaling
├── teachers.py            # CLIP + DINO teacher manager
├── students.py            # Modular backbone + FasterRCNN head + distill taps
├── loss.py                # KD loss (det + feature + logit)
utils/
├── logger.py          # TensorBoard logger
└── utils.py           # viz + helpers
train.py               # Full training pipeline
eval.py                # VOC mAP@[0.50:0.95] evaluation
```

### Student Detector Architecture:

```
StudentDetector
|
|-- Backbone (ResNet / MobileNet / ViT)
|-- BackboneWithHook → taps mid-level features
|-- FasterRCNN detection head
|-- Optional:
       |-- Feature projector (1×1 conv)
       |-- Logit classifier (global avgpool + FC)
```

---

# 🔬 Distillation Flow

### Feature KD (DINO)

* Extract teacher features
* Extract student projected features
* Compute MSE loss

### Logit KD (CLIP)

* Extract teacher logits
* Extract student auxiliary logits
* Compute KL-div loss (temperature-scaled)

### Total Loss

```
L_total = α * L_detection + β * L_featureKD + γ * L_logitKD
```

---

# 🐛 Debugging

A quick one-batch test exists:

```
debug_train.py
```

Run:

```bash
python debug_train.py
```

It prints:

* detection loss
* feature KD loss
* logit KD loss

---

# ✨ Example Experiment Plan

Full experiment grid (recommended):

| Backbone  | Distillation | β         | γ         | Notes          |
| --------- | ------------ | --------- | --------- | -------------- |
| ResNet18  | No KD        | 0         | 0         | Baseline       |
| MobileNet | No KD        | 0         | 0         | Baseline       |
| ViT Tiny  | No KD        | 0         | 0         | Baseline       |
| ResNet18  | Feature KD   | 1/5/10/20 | 0         | Ablation       |
| ResNet18  | Logit KD     | 0         | 0.5/1/2/5 | Ablation       |
| ResNet18  | Full KD      | best      | best      | Final          |
| MobileNet | Full KD      | best      | best      | Generalization |
| ViT Tiny  | Full KD      | best      | best      | Generalization |

---