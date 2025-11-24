import torch
from src.students import StudentDetector

def verify():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # 1. Initialize student model
    # Make sure model_type matches refactored names:
    # "resnet18", "mobilenet_v3_small", or "vit_tiny"
    student = StudentDetector(
        model_type="mobilenet_v3_small",
        teacher_feature_dim=384,
        use_feature_distill=True,
        use_logit_distill=True,
    ).to(device).eval()

    # 2. Dummy Input (2 images, 3×480×480)
    dummy_images = torch.randn(2, 3, 480, 480).to(device)

    # 3. Forward (inference mode; no targets)
    print("\nRunning forward pass...")
    detections, student_features, student_logits = student(dummy_images)

    # ---------------------------------------------
    # 4. Check detection outputs
    # ---------------------------------------------
    print("\n=== Detection Output Check ===")
    print(f"Detections type: {type(detections)}")

    if isinstance(detections, list):
        print(f"Number of prediction dicts: {len(detections)}")
        print(f"Keys in first prediction: {detections[0].keys()}")
        print(f"Boxes shape: {detections[0]['boxes'].shape}")
        print(f"Scores shape: {detections[0]['scores'].shape}")
        print(f"Labels shape: {detections[0]['labels'].shape}")
    else:
        print("ERROR: Expected a list of detection dicts.")

    # ---------------------------------------------
    # 5. Distillation feature check
    # ---------------------------------------------
    print("\n=== Distillation Feature Check (DINO) ===")
    if student_features is None:
        print("WARNING: student_features is None (feature distill disabled?)")
    else:
        print(f"Projected Feature Shape: {student_features.shape}")
        C = student_features.shape[1]

        if C == 384:
            print("SUCCESS: Feature channels correctly mapped to 384.")
        else:
            print(f"ERROR: Feature channels = {C}, expected 384.")

    # ---------------------------------------------
    # 6. Distillation logits check (CLIP-logit KD)
    # ---------------------------------------------
    print("\n=== Distillation Logit Check (CLIP) ===")
    if student_logits is None:
        print("WARNING: student_logits is None (logit distill disabled?)")
    else:
        print(f"Student Logit Shape: {student_logits.shape}")
        print("Expected shape = (batch_size, 20 VOC classes)")

        if student_logits.shape[1] == 20:
            print("SUCCESS: Student logits match expected VOC class dimension.")
        else:
            print(f"ERROR: Logit dim = {student_logits.shape[1]}, expected 20.")

    print("\nVerification complete.")

if __name__ == "__main__":
    verify()
