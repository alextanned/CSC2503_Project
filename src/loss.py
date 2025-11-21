import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, temperature=3.0):
        super().__init__()
        self.alpha = alpha # Weight for Detection Loss (Ground Truth)
        self.beta = beta   # Weight for Feature Distillation (DINO)
        self.mse = nn.MSELoss()

    def forward(self, student_loss_dict, student_features, teacher_features):
        """
        Combines Standard Detection Loss with Feature Distillation.
        """
        # 1. Standard Detection Loss (from SSDLite)
        # student_loss_dict contains 'bbox_regression' and 'classification' losses
        det_loss = sum(loss for loss in student_loss_dict.values())

        # 2. Feature Distillation (Student -> Teacher)
        # Student: (Batch, 384, 30, 30) -- usually
        # Teacher: (Batch, 384, 16, 16) -- usually
        
        # We interpolate Student to match Teacher's spatial size
        # We use 'bilinear' interpolation which is standard for feature maps
        if student_features is not None and teacher_features is not None:
            s_feat_resized = F.interpolate(
                student_features, 
                size=teacher_features.shape[-2:], # Target (H, W) 
                mode='bilinear', 
                align_corners=False
            )
            
            distill_loss = self.mse(s_feat_resized, teacher_features)
        else:
            distill_loss = torch.tensor(0.0, device=det_loss.device)

        # 3. Total Weighted Loss
        total_loss = (self.alpha * det_loss) + (self.beta * distill_loss)

        return total_loss, det_loss, distill_loss