#!/usr/bin/env python3
"""
Convert a trained ResNet18 model to FP16 (half precision) for efficient inference.
This script loads a trained FP32 model and converts it to FP16.

Usage:
    python quantize_model.py --checkpoint checkpoints/resnet18_baseline/student_latest.pth \
                             --output checkpoints/resnet18_baseline/student_fp16.pth
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
import os
from torch.utils.data import DataLoader
from src.students import StudentDetector
from src.dataset import DistillationVOCDataset
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import copy


def convert_to_fp16(model):
    """
    Convert model to FP16 (half precision).
    """
    print("Converting model to FP16...")
    model = model.half()
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert a trained ResNet18 detector to FP16")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained FP32 checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the FP16 model")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone architecture")
    parser.add_argument("--input-size", type=int, default=480,
                        help="Input image size")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FP16 Conversion for Object Detection")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Backbone: {args.backbone}")
    print()
    
    # Load the trained FP32 model
    print("Loading FP32 model...")
    
    # First, try to load checkpoint to check if it has distillation components
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Check if the checkpoint has distillation components
    has_distillation = any(k.startswith('projector.') or k.startswith('aux_classifier.') for k in state_dict.keys())
    
    print(f"Checkpoint has distillation components: {has_distillation}")
    
    model = StudentDetector(
        model_type=args.backbone,
        num_classes=20,  # VOC has 20 classes (background is added internally)
        use_feature_distill=has_distillation,
        use_logit_distill=has_distillation,
        input_size=args.input_size,
    )
    
    # Load checkpoint weights
    model.load_state_dict(state_dict)
    
    print("Model loaded successfully!")
    
    # Convert to FP16
    model = convert_to_fp16(model)
    model.eval()
    
    # Save the FP16 model
    print(f"\nSaving FP16 model to {args.output}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'fp16': True,
        'source_checkpoint': args.checkpoint,
    }, args.output)
    
    # Print model size comparison
    if os.path.exists(args.checkpoint):
        fp32_size = os.path.getsize(args.checkpoint) / (1024 * 1024)
        fp16_size = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\nModel size comparison:")
        print(f"  FP32: {fp32_size:.2f} MB")
        print(f"  FP16: {fp16_size:.2f} MB")
        print(f"  Compression ratio: {fp32_size / fp16_size:.2f}x")
    
    print("\n" + "=" * 60)
    print("FP16 Conversion complete!")
    print("=" * 60)
    print(f"\nTo evaluate the FP16 model, use:")
    print(f"  python eval.py --checkpoint {args.output} --backbone {args.backbone} --fp16")


if __name__ == "__main__":
    main()
