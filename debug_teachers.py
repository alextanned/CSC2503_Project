import torch
import matplotlib.pyplot as plt
from src.teachers import TeacherManager, VOC_CLASSES
from src.dataset import get_loaders

def verify():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # 1. Init Teachers
    teachers = TeacherManager(device=device)
    
    # 2. Get Data
    print("Loading one batch of data...")
    train_loader, _ = get_loaders(batch_size=2)
    images, _, targets = next(iter(train_loader))
    
    # Move images to GPU
    images = images.to(device)
    
    # 3. Run Teachers
    print("Running inference...")
    outputs = teachers(images)
    
    # 4. Analysis
    clip_logits = outputs['clip_logits']
    dino_feats = outputs['dino_features']
    
    print("\n--- CLIP Analysis ---")
    print(f"Logits Shape: {clip_logits.shape} (Should be Batch x 20)")
    
    # Show top 3 predictions for the first image
    probs = clip_logits[0].softmax(dim=-1)
    top_probs, top_idxs = probs.topk(3)
    
    print("Top 3 CLIP Predictions for Image 0:")
    for p, idx in zip(top_probs, top_idxs):
        print(f"  {VOC_CLASSES[idx]}: {p.item():.4f}")
        
    # Compare with Ground Truth
    gt_labels = targets[0]['labels']
    gt_names = [VOC_CLASSES[i] for i in gt_labels]
    print(f"Ground Truth Objects: {gt_names}")

    print("\n--- DINO Analysis ---")
    print(f"Feature Map Shape: {dino_feats.shape}")
    print(f"  (Batch: {dino_feats.shape[0]}, Channels: {dino_feats.shape[1]}, H: {dino_feats.shape[2]}, W: {dino_feats.shape[3]})")

    # Visualizing DINO Feature Norm (Attention Heatmap-ish)
    # Sum across channels to see where DINO is 'looking'
    heatmap = dino_feats[0].norm(dim=0).cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image (Normalized)")
    # Permute for matplotlib (C,H,W -> H,W,C) and move to cpu
    plt.imshow(images[0].cpu().permute(1, 2, 0)) 
    
    plt.subplot(1, 2, 2)
    plt.title("DINO Feature Activation")
    plt.imshow(heatmap, cmap='viridis')
    plt.show()

if __name__ == "__main__":
    verify()