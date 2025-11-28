#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Converting All ResNet18 Models to FP16"
echo "=========================================="

# Find all ResNet18 checkpoints
RESNET_CHECKPOINTS=$(find checkpoints -name "student_latest.pth" | grep -E "resnet18" | grep -v "fp16" | sort)

if [ -z "$RESNET_CHECKPOINTS" ]; then
    echo "Error: No ResNet18 checkpoints found!"
    echo "Please train the models first"
    exit 1
fi

echo "Found ResNet18 checkpoints:"
echo "$RESNET_CHECKPOINTS"
echo ""

# Convert each checkpoint to FP16
for CHECKPOINT in $RESNET_CHECKPOINTS; do
    # Extract directory name
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
    BASE_NAME=$(basename "$CHECKPOINT_DIR")
    
    # Create output path
    OUTPUT_PATH="${CHECKPOINT_DIR}/student_fp16.pth"
    
    echo "=========================================="
    echo "Converting: $BASE_NAME"
    echo "=========================================="
    echo "Input:  $CHECKPOINT"
    echo "Output: $OUTPUT_PATH"
    echo ""
    
    # Convert to FP16
    python quantize_model.py \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT_PATH" \
        --backbone resnet18 \
        --input-size 480
    
    echo ""
done

echo "=========================================="
echo "All Conversions Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "$RESNET_CHECKPOINTS" | while read CHECKPOINT; do
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
    echo "  - ${CHECKPOINT_DIR}/student_fp16.pth"
done
echo ""

echo "=========================================="
echo "Evaluating All FP16 Models"
echo "=========================================="
echo ""

# Evaluate each FP16 model
for CHECKPOINT in $RESNET_CHECKPOINTS; do
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
    BASE_NAME=$(basename "$CHECKPOINT_DIR")
    FP16_PATH="${CHECKPOINT_DIR}/student_fp16.pth"
    
    if [ -f "$FP16_PATH" ]; then
        echo "=========================================="
        echo "Evaluating: $BASE_NAME (FP16)"
        echo "=========================================="
        echo "Checkpoint: $FP16_PATH"
        echo ""
        
        python eval_performance.py \
            --backbone resnet18 \
            --checkpoint "$FP16_PATH" \
            --fp16 \
            --batch-size 8 \
            --device cuda \
            --input-size 480 \
            --num-timing-batches 50
        
        echo ""
    fi
done

echo "=========================================="
echo "All Evaluations Complete!"
echo "=========================================="
