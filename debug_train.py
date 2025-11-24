# debug_train_step.py
import torch

from src.dataset import get_loaders
from src.teachers import TeacherManager
from src.students import StudentDetector
from src.loss import DistillationLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # one small loader
    train_loader, _ = get_loaders(batch_size=2)

    teachers = TeacherManager(device=device)
    student = StudentDetector(
        model_type="resnet18",
        num_classes=20,
        teacher_feature_dim=384,
        use_feature_distill=True,
        use_logit_distill=True,
    ).to(device)

    criterion = DistillationLoss(alpha=1.0, beta=1.0, gamma=1.0)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    student.train()

    student_imgs, teacher_imgs, targets = next(iter(train_loader))
    student_imgs = student_imgs.to(device)
    teacher_imgs = teacher_imgs.to(device)

    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    with torch.no_grad():
        t_out = teachers(teacher_imgs)
        dino_features = t_out["dino_features"]
        clip_logits = t_out["clip_logits"]

    loss_dict, student_features, student_logits = student(student_imgs, targets=targets)

    total_loss, det_loss, feat_loss, logit_loss = criterion(
        loss_dict,
        student_features,
        dino_features,
        student_logits,
        clip_logits,
    )

    print("=== Debug Train Step ===")
    print(f"Detection loss: {det_loss.item():.4f}")
    print(f"Feature KD loss: {feat_loss.item():.4f}")
    print(f"Logit KD loss: {logit_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("Backward + optimizer.step() succeeded.")


if __name__ == "__main__":
    main()
