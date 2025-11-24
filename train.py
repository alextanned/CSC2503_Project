# train.py
import argparse
import os
import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_loaders
from src.teachers import TeacherManager
from src.students import StudentDetector
from src.loss import DistillationLoss
from utils.logger import ExperimentLogger
from utils.utils import plot_prediction_batch, plot_feature_heatmap
from eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train student detector with optional distillation on Pascal VOC")

    # core experiment knobs
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small", "vit_tiny"],
                        help="Student backbone architecture")
    parser.add_argument("--distill", action="store_true",
                        help="Enable knowledge distillation (DINO features + CLIP logits)")

    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for detection loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for feature distillation")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for logit distillation")

    # optimization
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # logging / paths
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional run name for TensorBoard/logs. If omitted, one is auto-generated.")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")

    # device
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g. 'cuda', 'cpu'). Default: auto-detect")
    
    parser.add_argument("--eval-every-epoch", action="store_true",
                    help="Run VOC mAP evaluation at the end of every epoch")
    parser.add_argument("--eval-final-only", action="store_true",
                        help="Run only final mAP evaluation after training")

    parser.add_argument("--input-size", type=int, default=480, help="Input image size (assumed square)")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode, on small subset of data")

    return parser.parse_args()


def train_one_epoch(
    epoch,
    student,
    teachers,
    loader,
    optimizer,
    loss_fn,
    logger,
    device,
    use_distill: bool,
    log_freq: int = 10,
    debug: bool = False,
):
    student.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")

    for i, (student_imgs, teacher_imgs, targets) in enumerate(pbar):
        step = epoch * len(loader) + i

        if debug and i > 50:
            break

        # move targets to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        student_imgs = student_imgs.to(device)

        # teacher images (if using distillation)
        if use_distill:
            if not isinstance(teacher_imgs, torch.Tensor) or teacher_imgs.ndim < 4:
                teacher_imgs = student_imgs
            teacher_imgs = teacher_imgs.to(device)

            with torch.no_grad():
                t_out = teachers(teacher_imgs)
                dino_features = t_out["dino_features"]
                clip_logits = t_out["clip_logits"]
        else:
            dino_features = None
            clip_logits = None

        # 1) student forward
        loss_dict, student_features, student_logits = student(student_imgs, targets=targets)

        # 2) total loss
        total_loss, det_loss, feat_loss, logit_loss = loss_fn(
            loss_dict,
            student_features,
            dino_features,
            student_logits,
            clip_logits,
        )

        # 3) optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        if i % log_freq == 0:
            metrics = {
                "Loss/Total": total_loss.item(),
                "Loss/Detection": det_loss.item(),
                "Loss/FeatureKD": feat_loss.item(),
                "Loss/LogitKD": logit_loss.item(),
            }
            logger.log_scalars(metrics, step)

        pbar.set_postfix(
            {
                "Total": f"{total_loss.item():.3f}",
                "Det": f"{det_loss.item():.3f}",
            }
        )

    return running_loss / len(loader)


@torch.no_grad()
def validate_and_visualize(
    epoch,
    student,
    teachers,
    loader,
    logger,
    device,
    use_distill: bool,
):
    student.eval()
    print("Running Validation & Visualization...")

    try:
        student_imgs, teacher_imgs, targets = next(iter(loader))
    except StopIteration:
        print("Validation loader is empty.")
        return

    student_imgs = student_imgs.to(device)

    if not isinstance(teacher_imgs, torch.Tensor) or teacher_imgs.ndim < 4:
        teacher_imgs = student_imgs
    teacher_imgs = teacher_imgs.to(device)

    # 1) student detections
    detections, student_features, _ = student(student_imgs)

    # 2) teacher features (for heatmap) if distill is used
    if use_distill and teachers is not None:
        t_out = teachers(teacher_imgs)
        teacher_features = t_out["dino_features"]
    else:
        teacher_features = None

    # visualization 1: prediction overlays
    viz_batch = plot_prediction_batch(student_imgs, targets, detections)
    logger.log_images("Qualitative/Predictions", viz_batch, epoch)

    # visualization 2: distillation feature heatmaps
    if teacher_features is not None and student_features is not None:
        heatmap_batch = plot_feature_heatmap(student_features, teacher_features)
        logger.log_images("Qualitative/Distillation_Heatmaps", heatmap_batch, epoch)


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # experiment name
    if args.run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        kd_tag = "KD" if args.distill else "baseline"
        args.run_name = f"{args.backbone}_{kd_tag}_a{args.alpha}_b{args.beta}_g{args.gamma}_{timestamp}"

    experiment_dir = os.path.join("runs", args.run_name)
    print(f"Starting Experiment: {experiment_dir}")
    logger = ExperimentLogger(log_dir=experiment_dir)

    # data
    train_loader, val_loader = get_loaders(batch_size=args.batch_size, input_size=args.input_size)

    # teachers (only if distillation enabled)
    teachers = TeacherManager(device=device) if args.distill else None

    # student
    student = StudentDetector(
        model_type=args.backbone,
        num_classes=20,
        teacher_feature_dim=384,
        use_feature_distill=args.distill,
        use_logit_distill=args.distill,
        input_size=args.input_size,
    ).to(device)

    # optimizer
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    # loss
    if args.distill:
        criterion = DistillationLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    else:
        # feature/logit KD disabled
        criterion = DistillationLoss(alpha=args.alpha, beta=0.0, gamma=0.0)

    # training loop

    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_one_epoch(
            epoch,
            student,
            teachers,
            train_loader,
            optimizer,
            criterion,
            logger,
            device=device,
            use_distill=args.distill,
            debug=args.debug,
        )

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")

        # Visualizations
        validate_and_visualize(
            epoch,
            student,
            teachers,
            val_loader,
            logger,
            device=device,
            use_distill=args.distill,
        )

        # Save checkpoint
        latest_path = os.path.join(args.output_dir, "student_latest.pth")
        torch.save(student.state_dict(), latest_path)

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f"student_epoch_{epoch+1}.pth")
            torch.save(student.state_dict(), ckpt_path)

        # ——— VOC mAP evaluation ———
        if args.eval_every_epoch and not args.eval_final_only:
            print("\nRunning VOC evaluation...")
            iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
            mAPs, overall = evaluate(student, val_loader, device=device, iou_thresholds=iou_thresholds, debug=args.debug)

            print(f"Epoch {epoch+1} VOC mAP@[0.50:0.95]: {overall:.4f}")
            logger.log_scalars({"Eval/mAP_all": overall}, epoch)

            for thr, ap in mAPs.items():
                logger.log_scalars({f"Eval/mAP@{thr:.2f}": ap}, epoch)

    if args.eval_final_only or not args.eval_every_epoch:
        print("\nRunning final VOC evaluation...")
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        mAPs, overall = evaluate(student, val_loader, device=device, iou_thresholds=iou_thresholds)

        print("\n===== FINAL VOC mAP RESULTS =====")
        for thr, val in mAPs.items():
            print(f"mAP@IoU={thr:.2f}: {val:.4f}")
        print(f"\nFinal Overall mAP@[0.50:0.95]: {overall:.4f}")

        logger.log_scalars({"EvalFinal/mAP_all": overall}, 0)

    logger.close()


if __name__ == "__main__":
    main()
