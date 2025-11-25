import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.dataset import VOC_CLASSES

# Color map for bounding boxes
COLORS = np.random.uniform(0, 255, size=(len(VOC_CLASSES), 3))

def tensor_to_numpy(img_tensor):
    """
    Converts (C, H, W) Tensor to (H, W, C) Numpy for plotting.
    Un-normalizes if necessary (assuming inputs are 0-1).
    """
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    # Convert to uint8 for OpenCV
    img = (img * 255).astype(np.uint8)
    return img.copy()  # Copy required for cv2 drawing

def plot_prediction_batch(
    images,
    targets,
    predictions=None,
    limit=4,
    score_thresh=0.3,
):
    """
    Draws Ground Truth (Green) and Predictions (Red) on images.
    Assumes:
      - GT labels: 1..20  (0 is background, not used)
      - Pred labels: 0..20 (0 = background, 1..20 = classes)
      - VOC_CLASSES indexed 0..19
    Returns: (N, 3, H, W) tensor for logging/saving.
    """
    canvas_list = []
    n = min(len(images), limit)
    
    for i in range(n):
        img = tensor_to_numpy(images[i])
        
        # 1. Draw Ground Truth (Green)
        if targets is not None:
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            gt_labels = targets[i]['labels'].cpu().numpy()
            
            for box, label_idx in zip(gt_boxes, gt_labels):
                if label_idx <= 0:
                    continue
                cls_idx = int(label_idx) - 1  # map 1..20 -> 0..19
                cls_name = VOC_CLASSES[cls_idx] if 0 <= cls_idx < len(VOC_CLASSES) else "Bg"

                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    cls_name,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # 2. Draw Predictions (Red) if provided
        if predictions is not None:
            pred_boxes = predictions[i]['boxes'].detach().cpu().numpy()
            pred_scores = predictions[i]['scores'].detach().cpu().numpy()
            pred_labels = predictions[i]['labels'].detach().cpu().numpy()
            
            keep = pred_scores > score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            for box, label_idx, score in zip(pred_boxes, pred_labels, pred_scores):
                if label_idx <= 0:
                    continue  # skip background
                cls_idx = int(label_idx) - 1  # 1..20 -> 0..19
                cls_name = VOC_CLASSES[cls_idx] if 0 <= cls_idx < len(VOC_CLASSES) else "Bg"

                x1, y1, x2, y2 = box.astype(int)
                label_idx = label_idx - 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"{cls_name}: {score:.2f}",
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
        
        canvas_list.append(torch.from_numpy(img).permute(2, 0, 1))
        
    return torch.stack(canvas_list)

def plot_feature_heatmap(student_features, teacher_features, limit=4):
    """
    Visualizes the 'Attention' of Student vs Teacher.
    """
    # Create figure
    fig, axs = plt.subplots(limit, 2, figsize=(8, 4*limit))
    
    # Handle single sample case (limit=1) where axs is 1D
    if limit == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(limit):
        # Teacher Heatmap
        t_map = teacher_features[i].mean(dim=0).cpu().numpy()
        t_map = (t_map - t_map.min()) / (t_map.max() - t_map.min() + 1e-8) 
        
        # Student Heatmap
        s_map = student_features[i].mean(dim=0).detach().cpu().numpy()
        s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min() + 1e-8)
        
        # Plot
        ax_t, ax_s = axs[i, 0], axs[i, 1]
            
        ax_t.imshow(t_map, cmap='jet')
        ax_t.set_title("Teacher Attention (DINO)")
        ax_t.axis('off')
        
        ax_s.imshow(s_map, cmap='jet')
        ax_s.set_title("Student Attention")
        ax_s.axis('off')
        
    plt.tight_layout()
    
    fig.canvas.draw()
    img_np = np.array(fig.canvas.renderer.buffer_rgba())
    img_np = img_np[:, :, :3]
    
    plt.close(fig)
    
    return torch.from_numpy(img_np.copy()).permute(2, 0, 1).unsqueeze(0)