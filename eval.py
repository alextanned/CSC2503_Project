# eval_voc.py
import argparse
import os
import numpy as np
import torch
from torchvision.ops import box_iou
import tqdm

from src.dataset import get_loaders, VOC_CLASSES
from src.students import StudentDetector
from src.teacher_detector import TeacherDetector


def voc_ap(rec, prec):
    """
    Continuous AP (area under interpolated precision-recall curve).
    rec, prec: 1D torch tensors
    """
    # Add boundary points
    mrec = torch.cat([torch.tensor([0.0]), rec, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), prec, torch.tensor([0.0])])

    # Make precision monotonically decreasing
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.max(mpre[i], mpre[i + 1])

    # Integrate area where recall changes
    idx = torch.nonzero(mrec[1:] != mrec[:-1]).squeeze()
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap.item()


def compute_ap_per_class(all_detections, all_annotations, num_classes, iou_thr=0.5, device="cpu"):
    """
    all_detections / all_annotations: lists of length N_images, each a dict:
      - boxes: (N_i, 4) tensor
      - labels: (N_i,) tensor  in [0, num_classes-1]
      - scores: (N_i,) tensor  (detections only)
    """
    aps = []

    for cls_id in range(num_classes):
        # gather gt boxes per image
        gt_by_img = {}
        total_gts = 0

        for img_idx, ann in enumerate(all_annotations):
            labels = ann["labels"]
            boxes = ann["boxes"]
            mask = labels == cls_id
            gt_boxes = boxes[mask].to(device)

            gt_by_img[img_idx] = {
                "boxes": gt_boxes,
                "detected": torch.zeros(len(gt_boxes), dtype=torch.bool, device=device),
            }
            total_gts += len(gt_boxes)

        if total_gts == 0:
            aps.append(float("nan"))
            continue

        # collect predictions for this class
        preds = []
        for img_idx, det in enumerate(all_detections):
            labels = det["labels"]
            boxes = det["boxes"]
            scores = det["scores"]
            mask = labels == cls_id
            for b, s in zip(boxes[mask], scores[mask]):
                preds.append((img_idx, s.item(), b.to(device)))

        if len(preds) == 0:
            aps.append(0.0)
            continue

        # sort by score descending
        preds.sort(key=lambda x: x[1], reverse=True)

        tp = []
        fp = []

        for img_idx, score, box in preds:
            gt_info = gt_by_img[img_idx]
            gt_boxes = gt_info["boxes"]
            detected = gt_info["detected"]

            if gt_boxes.numel() == 0:
                fp.append(1)
                tp.append(0)
                continue

            ious = box_iou(box.unsqueeze(0), gt_boxes)[0]  # (num_gt,)
            best_iou, best_idx = ious.max(0)

            if best_iou >= iou_thr and not detected[best_idx]:
                tp.append(1)
                fp.append(0)
                detected[best_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        tp = torch.tensor(tp, dtype=torch.float32, device=device)
        fp = torch.tensor(fp, dtype=torch.float32, device=device)

        tp_cum = torch.cumsum(tp, 0)
        fp_cum = torch.cumsum(fp, 0)

        rec = tp_cum / total_gts
        prec = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)

        aps.append(voc_ap(rec.cpu(), prec.cpu()))

    return aps


def evaluate(model, val_loader, device="cuda", iou_thresholds=None, debug=False):
    model.eval()
    iou_thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]

    all_detections = []
    all_annotations = []

    pbar = tqdm.tqdm(val_loader, desc="Evaluating")

    with torch.no_grad():
        for i, (student_imgs, teacher_imgs, targets) in tqdm.tqdm(enumerate(pbar)):
            if debug and i > 50:
                break

            student_imgs = student_imgs.to(device)
            outputs, _, _ = model(student_imgs)

            for o, t in zip(outputs, targets):
                # predictions from FasterRCNN:
                # labels are [0..num_classes], with 0 = background
                pred_boxes = o["boxes"].to(device)
                pred_scores = o["scores"].to(device)
                pred_labels = o["labels"].to(device)

                # drop background and shift to 0-based class indices
                fg_mask = pred_labels > 0
                pred_boxes = pred_boxes[fg_mask]
                pred_scores = pred_scores[fg_mask]
                pred_labels = pred_labels[fg_mask] - 1  # now in [0, 19]

                all_detections.append(
                    {
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels,
                    }
                )

                all_annotations.append(
                    {
                        "boxes": t["boxes"].to(device),
                        "labels": t["labels"].to(device) - 1,  # shift to 0-based class indices
                    }
                )

    num_classes = len(VOC_CLASSES)
    mAPs = {}

    for thr in iou_thresholds:
        aps = compute_ap_per_class(all_detections, all_annotations, num_classes, iou_thr=thr, device=device)
        cls_mean_ap = np.nanmean(aps)
        mAPs[thr] = cls_mean_ap

    # overall mAP across IoUs 0.5:0.95
    overall = float(np.mean(list(mAPs.values())))
    return mAPs, overall


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VOC mAP for a trained student detector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to student .pth checkpoint")
    parser.add_argument("--eval-teacher-detector", action="store_true",
                        help="Evaluate a teacher detector (DINO/CLIP with detection head)")
    parser.add_argument("--teacher-type", type=str, default="dino",
                        choices=["dino", "clip", "both"],
                        help="Teacher backbone type (for --eval-teacher-detector)")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small", "vit_tiny"],
                        help="Backbone architecture used in the student model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., 'cuda', 'cpu'); default auto-detect")
    parser.add_argument("--input-size", type=int, default=480, help="Input image size (assumed square)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode, on small subset of data")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data (we only need val loader)
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
    else:
        print(f"Loading student detector ({args.backbone})...")
        # student model (no distillation taps needed during eval, but they don't hurt)
        model = StudentDetector(
            model_type=args.backbone,
            num_classes=20,
            teacher_feature_dim=384,
            use_feature_distill=False,
            use_logit_distill=False,
            input_size=args.input_size,
        ).to(device)

    # load weights
    assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # evaluate
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    mAPs, overall = evaluate(model, val_loader, device=device, iou_thresholds=iou_thresholds, debug=args.debug)

    print("\n===== VOC mAP Results =====")
    for thr, val in mAPs.items():
        print(f"mAP@IoU={thr:.2f}: {val:.4f}")
    print(f"\nOverall mAP@[0.50:0.95]: {overall:.4f}")


if __name__ == "__main__":
    main()
