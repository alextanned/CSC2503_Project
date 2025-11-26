# eval_performance.py
import argparse
import os
import time
import numpy as np
import torch

from src.dataset import get_loaders
from src.students import StudentDetector
from src.teacher_detector import TeacherDetector
from eval import evaluate


def measure_inference_time(model, val_loader, device="cuda", warmup_batches=5, num_batches=50):
    """Measure average inference time per image."""
    model.eval()
    
    times = []
    total_images = 0
    
    # Check if model is in FP16
    is_fp16 = next(model.parameters()).dtype == torch.float16
    
    with torch.no_grad():
        # Warmup
        for i, (student_imgs, _, _) in enumerate(val_loader):
            if i >= warmup_batches:
                break
            student_imgs = student_imgs.to(device)
            if is_fp16:
                student_imgs = student_imgs.half()
            _ = model(student_imgs)
        
        # Actual timing
        for i, (student_imgs, _, _) in enumerate(val_loader):
            if i >= num_batches:
                break
                
            student_imgs = student_imgs.to(device)
            if is_fp16:
                student_imgs = student_imgs.half()
            batch_size = student_imgs.shape[0]
            
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.perf_counter()
            
            _ = model(student_imgs)
            
            torch.cuda.synchronize() if device == "cuda" else None
            end = time.perf_counter()
            
            times.append(end - start)
            total_images += batch_size
    
    avg_time_per_batch = np.mean(times)
    avg_time_per_image = np.sum(times) / total_images
    fps = total_images / np.sum(times)
    
    return {
        "avg_time_per_batch_ms": avg_time_per_batch * 1000,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "fps": fps,
        "total_images": total_images,
    }


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VOC mAP and runtime for a trained detector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--eval-teacher-detector", action="store_true",
                        help="Evaluate a teacher detector (DINO/CLIP with detection head)")
    parser.add_argument("--teacher-type", type=str, default="dino",
                        choices=["dino", "clip", "both"],
                        help="Teacher backbone type (for --eval-teacher-detector)")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small", "vit_tiny"],
                        help="Backbone architecture used in the student model")
    parser.add_argument("--fp16", action="store_true",
                        help="Load a FP16 (half precision) model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., 'cuda', 'cpu'); default auto-detect")
    parser.add_argument("--num-timing-batches", type=int, default=50,
                        help="Number of batches to use for timing measurements")
    parser.add_argument("--input-size", type=int, default=480, help="Input image size")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    _, val_loader = get_loaders(batch_size=args.batch_size, input_size=args.input_size)

    # Load model based on type
    if args.eval_teacher_detector:
        print(f"Loading teacher detector ({args.teacher_type})...")
        model = TeacherDetector(
            teacher_type=args.teacher_type,
            num_classes=20,
            freeze_backbone=True,
            input_size=args.input_size,
            device=device
        ).to(device)
        model_name = f"teacher_{args.teacher_type}"
    else:
        print(f"Loading student detector ({args.backbone})...")
        model = StudentDetector(
            model_type=args.backbone,
            num_classes=20,
            teacher_feature_dim=384,
            use_feature_distill=False,
            use_logit_distill=False,
            input_size=args.input_size,
        ).to(device)
        model_name = args.backbone

    assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Model size
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)")
    print(f"{'='*60}\n")

    # Performance metrics
    print("Measuring inference time...")
    perf_metrics = measure_inference_time(
        model, val_loader, device=device, num_batches=args.num_timing_batches
    )

    print(f"\n{'='*60}")
    print("RUNTIME PERFORMANCE")
    print(f"{'='*60}")
    print(f"Average time per batch: {perf_metrics['avg_time_per_batch_ms']:.2f} ms")
    print(f"Average time per image: {perf_metrics['avg_time_per_image_ms']:.2f} ms")
    print(f"Throughput (FPS): {perf_metrics['fps']:.2f}")
    print(f"Tested on {perf_metrics['total_images']} images")
    print(f"{'='*60}\n")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Parameters: {total_params:,}")
    print(f"Inference Speed: {perf_metrics['avg_time_per_image_ms']:.2f} ms/image ({perf_metrics['fps']:.2f} FPS)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
