# eval_performance.py
import argparse
import os
import time
import numpy as np
import torch

from src.dataset import get_loaders
from src.students import StudentDetector
from eval import evaluate


def measure_inference_time(model, val_loader, device="cuda", warmup_batches=5, num_batches=50):
    """Measure average inference time per image."""
    model.eval()
    
    times = []
    total_images = 0
    
    with torch.no_grad():
        # Warmup
        for i, (student_imgs, _, _) in enumerate(val_loader):
            if i >= warmup_batches:
                break
            student_imgs = student_imgs.to(device)
            _ = model(student_imgs)
        
        # Actual timing
        for i, (student_imgs, _, _) in enumerate(val_loader):
            if i >= num_batches:
                break
                
            student_imgs = student_imgs.to(device)
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
    parser = argparse.ArgumentParser(description="Evaluate VOC mAP and runtime for a trained student detector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to student .pth checkpoint")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small", "vit_tiny"],
                        help="Backbone architecture used in the student model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., 'cuda', 'cpu'); default auto-detect")
    parser.add_argument("--num-timing-batches", type=int, default=50,
                        help="Number of batches to use for timing measurements")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    _, val_loader = get_loaders(batch_size=args.batch_size)

    # Load model
    model = StudentDetector(
        model_type=args.backbone,
        num_classes=20,
        teacher_feature_dim=384,
        use_feature_distill=False,
        use_logit_distill=False,
    ).to(device)

    assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Model size
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Model: {args.backbone}")
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

    # mAP evaluation
    print("Evaluating mAP...")
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    mAPs, overall = evaluate(model, val_loader, device=device, iou_thresholds=iou_thresholds)

    print(f"\n{'='*60}")
    print("VOC mAP RESULTS")
    print(f"{'='*60}")
    for thr, val in mAPs.items():
        print(f"mAP@IoU={thr:.2f}: {val:.4f}")
    print(f"\nOverall mAP@[0.50:0.95]: {overall:.4f}")
    print(f"{'='*60}\n")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Backbone: {args.backbone}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Parameters: {total_params:,}")
    print(f"mAP@[0.50:0.95]: {overall:.4f}")
    print(f"Inference Speed: {perf_metrics['avg_time_per_image_ms']:.2f} ms/image ({perf_metrics['fps']:.2f} FPS)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
