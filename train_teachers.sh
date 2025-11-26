#!/bin/bash

# Train teacher detector models using DINO and CLIP backbones
# Run with: ./train_teachers.sh

set -e  # Exit on error

echo "=========================================="
echo "Training Teacher Detector Models"
echo "=========================================="

# Configuration
BATCH_SIZE=8
EPOCHS=10
LR=1e-3
INPUT_SIZE=480
DEVICE="cuda"

# Create output directory
mkdir -p checkpoints
mkdir -p runs

echo ""
echo "=========================================="
echo "1. Training DINO-based Teacher Detector"
echo "=========================================="
python train.py \
    --train-teacher-detector \
    --teacher-type dino \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --input-size ${INPUT_SIZE} \
    --device ${DEVICE} \
    --eval-final-only

echo ""
echo "=========================================="
echo "2. Training CLIP-based Teacher Detector"
echo "=========================================="
python train.py \
    --train-teacher-detector \
    --teacher-type clip \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --input-size ${INPUT_SIZE} \
    --device ${DEVICE} \
    --eval-final-only

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved in: checkpoints/"
echo "TensorBoard logs in: runs/"
echo ""
echo "To evaluate the models, run:"
echo "  python eval.py --eval-teacher-detector --teacher-type dino --checkpoint checkpoints/teacher_dino_*/teacher_latest.pth"
echo "  python eval.py --eval-teacher-detector --teacher-type clip --checkpoint checkpoints/teacher_clip_*/teacher_latest.pth"
