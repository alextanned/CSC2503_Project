import torch
import torch.optim as optim
from tqdm import tqdm
import os
import datetime

# Import modules
from src.dataset import get_loaders
from src.teachers import TeacherManager
from src.students import StudentDetector
from src.loss import DistillationLoss
from utils.logger import ExperimentLogger
from utils.utils import plot_prediction_batch, plot_feature_heatmap

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 30
ALPHA = 1.0   # Detection Weight
BETA = 10.0   # Distillation Weight (Needs to be higher usually)
LOG_FREQ = 10 # Log scalars every N batches

def train_one_epoch(epoch, student, teachers, loader, optimizer, loss_fn, logger):
    student.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    
    for i, (student_imgs, teacher_imgs, targets) in enumerate(pbar):
        step = epoch * len(loader) + i
        
        # Move to GPU
        student_imgs = student_imgs.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        # Fallback if teacher_imgs are weird
        if not isinstance(teacher_imgs, torch.Tensor):
            teacher_imgs = student_imgs
        teacher_imgs = teacher_imgs.to(DEVICE)

        # 1. Teacher Pass
        with torch.no_grad():
            teacher_out = teachers(teacher_imgs)
            dino_features = teacher_out['dino_features']

        # 2. Student Pass
        loss_dict, student_features = student(student_imgs, targets=targets)

        # 3. Loss Calculation
        total_loss, det_loss, dist_loss = loss_fn(loss_dict, student_features, dino_features)

        # 4. Optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 5. Logging
        running_loss += total_loss.item()
        
        if i % LOG_FREQ == 0:
            # Log detailed loss components
            metrics = {
                "Loss/Total": total_loss.item(),
                "Loss/Detection": det_loss.item(),
                "Loss/Distillation": dist_loss.item(),
                "Components/BBox_Reg": loss_dict.get('bbox_regression', 0),
                "Components/Cls": loss_dict.get('classification', 0)
            }
            logger.log_scalars(metrics, step)

        pbar.set_postfix({'Total': f"{total_loss.item():.2f}", 'Det': f"{det_loss.item():.2f}"})

    return running_loss / len(loader)

def validate_and_visualize(epoch, student, teachers, loader, logger):
    """
    Runs inference on a few batches to visualize bounding boxes and heatmaps.
    """
    student.eval()
    print("Running Validation & Visualization...")
    
    # Get just one batch for visualization
    student_imgs, teacher_imgs, targets = next(iter(loader))
    student_imgs = student_imgs.to(DEVICE)
    
    # Fallback
    if not isinstance(teacher_imgs, torch.Tensor):
        teacher_imgs = student_imgs
    teacher_imgs = teacher_imgs.to(DEVICE)

    with torch.no_grad():
        # 1. Get Predictions
        predictions, student_features = student(student_imgs) # No targets passed = Inference Mode
        
        # 2. Get Teacher Features for comparison
        teacher_out = teachers(teacher_imgs)
        teacher_features = teacher_out['dino_features']

    # --- Visualization 1: Bounding Boxes ---
    # Draw Ground Truth vs Student Predictions
    viz_batch = plot_prediction_batch(student_imgs, targets, predictions)
    logger.log_images("Qualitative/Predictions", viz_batch, epoch)

    # --- Visualization 2: Feature Heatmaps ---
    # Compare Student Feature Map vs Teacher Feature Map
    heatmap_batch = plot_feature_heatmap(student_features, teacher_features)
    logger.log_images("Qualitative/Distillation_Heatmaps", heatmap_batch, epoch)

def main():
    # 1. Setup Logger
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = f"runs/MobileNet_DINO_{timestamp}"
    
    print(f"Starting Experiment: {experiment_dir}")
    
    # Pass this unique path to the logger
    logger = ExperimentLogger(log_dir=experiment_dir)
    
    train_loader, val_loader = get_loaders(batch_size=BATCH_SIZE)
    
    teachers = TeacherManager(device=DEVICE)
    student = StudentDetector(model_type="mobilenet").to(DEVICE)
    
    optimizer = optim.Adam(student.parameters(), lr=LR)
    criterion = DistillationLoss(alpha=ALPHA, beta=BETA)
    
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(epoch, student, teachers, train_loader, optimizer, criterion, logger)
        
        # Validate & Visualize (Log images)
        validate_and_visualize(epoch, student, teachers, val_loader, logger)
        
        # Save Checkpoint
        torch.save(student.state_dict(), f"checkpoints/student_latest.pth")
        if (epoch+1) % 5 == 0:
            torch.save(student.state_dict(), f"checkpoints/student_epoch_{epoch+1}.pth")

    logger.close()

if __name__ == "__main__":
    main()