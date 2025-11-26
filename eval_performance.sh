#!/bin/bash

# Evaluate performance of teacher and student detector models
# Run with: ./eval_performance.sh

set -e  # Exit on error

echo "=========================================="
echo "Evaluating All Detector Performance"
echo "=========================================="

# Configuration
BATCH_SIZE=8
DEVICE="cuda"
INPUT_SIZE=480

# Find the latest checkpoints
DINO_CHECKPOINT=$(ls -t checkpoints/teacher_dino_*/teacher_latest.pth 2>/dev/null | head -1)
CLIP_CHECKPOINT=$(ls -t checkpoints/teacher_clip_*/teacher_latest.pth 2>/dev/null | head -1)
RESNET_CHECKPOINT=$(ls -t checkpoints/resnet18_baseline/student_latest.pth 2>/dev/null | head -1)
RESNET_FP16_CHECKPOINT=$(ls -t checkpoints/resnet18_baseline/student_fp16.pth 2>/dev/null | head -1)
MOBILENET_CHECKPOINT=$(ls -t checkpoints/mobilenet_baseline/student_latest.pth 2>/dev/null | head -1)
VIT_CHECKPOINT=$(ls -t checkpoints/vit_tiny_*/student_latest.pth 2>/dev/null | head -1)

if [ -z "$DINO_CHECKPOINT" ] && [ -z "$CLIP_CHECKPOINT" ] && [ -z "$RESNET_CHECKPOINT" ] && [ -z "$MOBILENET_CHECKPOINT" ] && [ -z "$VIT_CHECKPOINT" ]; then
    echo "Error: No checkpoints found!"
    echo "Please train the models first"
    exit 1
fi

# Evaluate Teacher Detectors
echo ""
echo "=========================================="
echo "TEACHER DETECTORS"
echo "=========================================="

# Evaluate DINO-based Teacher Detector
if [ -n "$DINO_CHECKPOINT" ]; then
    echo ""
    echo "=========================================="
    echo "1. Evaluating DINO-based Teacher Detector"
    echo "=========================================="
    echo "Checkpoint: $DINO_CHECKPOINT"
    echo ""
    
    python eval_performance.py \
        --eval-teacher-detector \
        --teacher-type dino \
        --checkpoint "$DINO_CHECKPOINT" \
        --batch-size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --input-size ${INPUT_SIZE} \
        --num-timing-batches 50
else
    echo ""
    echo "Skipping DINO: No checkpoint found"
fi

# Evaluate CLIP-based Teacher Detector
if [ -n "$CLIP_CHECKPOINT" ]; then
    echo ""
    echo "=========================================="
    echo "2. Evaluating CLIP-based Teacher Detector"
    echo "=========================================="
    echo "Checkpoint: $CLIP_CHECKPOINT"
    echo ""
    
    python eval_performance.py \
        --eval-teacher-detector \
        --teacher-type clip \
        --checkpoint "$CLIP_CHECKPOINT" \
        --batch-size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --input-size ${INPUT_SIZE} \
        --num-timing-batches 50
else
    echo ""
    echo "Skipping CLIP: No checkpoint found"
fi

# Evaluate Student Detectors
echo ""
echo "=========================================="
echo "STUDENT DETECTORS"
echo "=========================================="

# Evaluate ResNet18 Baseline
if [ -n "$RESNET_CHECKPOINT" ]; then
    echo ""
    echo "=========================================="
    echo "3. Evaluating ResNet18 Baseline"
    echo "=========================================="
    echo "Checkpoint: $RESNET_CHECKPOINT"
    echo ""
    
    python eval_performance.py \
        --backbone resnet18 \
        --checkpoint "$RESNET_CHECKPOINT" \
        --batch-size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --input-size ${INPUT_SIZE} \
        --num-timing-batches 50
else
    echo ""
    echo "Skipping ResNet18: No checkpoint found"
fi

# Evaluate ResNet18 FP16
if [ -n "$RESNET_FP16_CHECKPOINT" ]; then
    echo ""
    echo "=========================================="
    echo "4. Evaluating ResNet18 FP16 (Half Precision)"
    echo "=========================================="
    echo "Checkpoint: $RESNET_FP16_CHECKPOINT"
    echo ""
    
    python eval_performance.py \
        --backbone resnet18 \
        --checkpoint "$RESNET_FP16_CHECKPOINT" \
        --fp16 \
        --batch-size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --input-size ${INPUT_SIZE} \
        --num-timing-batches 50
else
    echo ""
    echo "Skipping ResNet18 FP16: No checkpoint found"
fi

# Evaluate MobileNet Baseline
if [ -n "$MOBILENET_CHECKPOINT" ]; then
    echo ""
    echo "=========================================="
    echo "5. Evaluating MobileNet Baseline"
    echo "=========================================="
    echo "Checkpoint: $MOBILENET_CHECKPOINT"
    echo ""
    
    python eval_performance.py \
        --backbone mobilenet_v3_small \
        --checkpoint "$MOBILENET_CHECKPOINT" \
        --batch-size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --input-size ${INPUT_SIZE} \
        --num-timing-batches 50
else
    echo ""
    echo "Skipping MobileNet: No checkpoint found"
fi

# Evaluate ViT Tiny
if [ -n "$VIT_CHECKPOINT" ]; then
    echo ""
    echo "=========================================="
    echo "6. Evaluating ViT Tiny"
    echo "=========================================="
    echo "Checkpoint: $VIT_CHECKPOINT"
    echo ""
    
    python eval_performance.py \
        --backbone vit_tiny \
        --checkpoint "$VIT_CHECKPOINT" \
        --batch-size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --input-size ${INPUT_SIZE} \
        --num-timing-batches 50
else
    echo ""
    echo "Skipping ViT: No checkpoint found"
fi

echo ""
echo "=========================================="
echo "Performance Evaluation Complete!"
echo "=========================================="
