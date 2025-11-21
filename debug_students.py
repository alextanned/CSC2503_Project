import torch
from src.students import StudentDetector

def verify():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # 1. Initialize Student
    # teacher_feature_dim=384 matches the ViT-Small DINO we used in Phase 2
    student = StudentDetector(model_type="mobilenet", teacher_feature_dim=384).to(device).eval()
    
    # 2. Create Dummy Data (Batch 2, 3 channels, 480x480)
    dummy_images = torch.randn(2, 3, 480, 480).to(device)
    
    # 3. Run Forward Pass (No targets = Inference Mode)
    print("Running Forward Pass...")
    output, features = student(dummy_images)
    
    # 4. Check Detection Output
    print(f"Detection Output Type: {type(output)}")
    # In inference, output is a list of dicts
    print(f"Number of predictions: {len(output)}")
    if len(output) > 0:
        print(f"First image boxes: {output[0]['boxes'].shape}")
        
    # 5. Check Distillation Features
    print("\n--- Distillation Feature Check ---")
    print(f"Projected Feature Shape: {features.shape}")
    print("Requirements for DINO alignment:")
    print("1. Channels must be 384 (to match DINO).")
    print("2. Spatial dim (H,W) should be somewhat related to image size.")
    
    if features.shape[1] == 384:
        print("SUCCESS: Feature channels aligned with Teacher.")
    else:
        print(f"ERROR: Channels are {features.shape[1]}, expected 384.")

if __name__ == "__main__":
    verify()