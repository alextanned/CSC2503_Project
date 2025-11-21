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
    return img.copy() # Copy required for cv2 drawing

def plot_prediction_batch(images, targets, predictions=None, limit=4):
    """
    Draws Ground Truth (Green) and Predictions (Red) on images.
    """
    canvas_list = []
    
    # Loop through batch
    for i in range(min(len(images), limit)):
        img = tensor_to_numpy(images[i])
        
        # 1. Draw Ground Truth (Green)
        if targets is not None:
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            gt_labels = targets[i]['labels'].cpu().numpy()
            
            for box, label_idx in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = VOC_CLASSES[label_idx] if label_idx < len(VOC_CLASSES) else "Bg"
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. Draw Predictions (Red) if provided
        if predictions is not None:
            pred_boxes = predictions[i]['boxes'].detach().cpu().numpy()
            pred_scores = predictions[i]['scores'].detach().cpu().numpy()
            pred_labels = predictions[i]['labels'].detach().cpu().numpy()
            
            # Filter by confidence
            keep = pred_scores > 0.3
            for box, label_idx, score in zip(pred_boxes[keep], pred_labels[keep], pred_scores[keep]):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{VOC_CLASSES[label_idx]}: {score:.2f}"
                cv2.putText(img, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Convert back to Channel First for TensorBoard
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