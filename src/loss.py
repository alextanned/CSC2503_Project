import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combines:
      - detection loss from torchvision model (sum of dict)
      - optional feature distillation (MSE)
      - optional logit distillation (KL)
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, temperature=3.0):
        super().__init__()
        self.alpha = alpha  # detection
        self.beta = beta    # feature KD (DINO)
        self.gamma = gamma  # logit KD (CLIP)
        self.T = temperature

        self.mse = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(
        self,
        student_loss_dict,
        student_features,
        teacher_features,
        student_logits,
        teacher_logits,
    ):
        # 1. detection loss: torchvision detection models return a dict of scalar losses
        det_loss = sum(student_loss_dict.values())

        # 2. feature distillation
        if self.beta != 0.0 and student_features is not None and teacher_features is not None:
            # resize student to teacher spatial size if needed
            if student_features.shape[-2:] != teacher_features.shape[-2:]:
                s_feat = F.interpolate(
                    student_features,
                    size=teacher_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                s_feat = student_features
            distill_feat_loss = self.mse(s_feat, teacher_features)
        else:
            distill_feat_loss = torch.zeros(1, device=det_loss.device)

        # 3. logit distillation
        if (
            self.gamma != 0.0
            and student_logits is not None
            and teacher_logits is not None
        ):
            # soften
            s_log_soft = F.log_softmax(student_logits / self.T, dim=1)
            t_soft = F.softmax(teacher_logits / self.T, dim=1)
            distill_logit_loss = (self.T ** 2) * self.kl_div(s_log_soft, t_soft)
        else:
            distill_logit_loss = torch.zeros(1, device=det_loss.device)

        total_loss = (
            self.alpha * det_loss
            + self.beta * distill_feat_loss
            + self.gamma * distill_logit_loss
        )

        return total_loss, det_loss, distill_feat_loss, distill_logit_loss
