#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Converting ResNet18 Baseline to FP16"
echo "=========================================="

# Check if checkpoint exists
CHECKPOINT="checkpoints/resnet18_baseline/student_latest.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please train the model first with:"
    echo "  python train.py --backbone resnet18 --epochs 10 --batch-size 8 --run-name resnet18_baseline"
    exit 1
fi

# Convert the model to FP16
python quantize_model.py \
    --checkpoint "$CHECKPOINT" \
    --output checkpoints/resnet18_baseline/student_fp16.pth \
    --backbone resnet18 \
    --input-size 480

echo ""
echo "=========================================="
echo "FP16 Conversion Complete!"
echo "=========================================="
echo "FP16 model saved to: checkpoints/resnet18_baseline/student_fp16.pth"

echo ""
echo "=========================================="
echo "Evaluating FP16 Model"
echo "=========================================="
python eval.py \
    --checkpoint checkpoints/resnet18_baseline/student_fp16.pth \
    --backbone resnet18 \
    --fp16

echo ""
echo "Done!"
