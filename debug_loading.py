import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.dataset import get_loaders, VOC_CLASSES

def visualize_batch():
    train_loader, _ = get_loaders()
    
    # Get one batch
    student_imgs, _, targets = next(iter(train_loader))
    
    # Plot the first image in the batch
    img_tensor = student_imgs[0]
    target = targets[0]
    
    # Convert tensor (C, H, W) to numpy (H, W, C)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)
    
    # Draw boxes
    for box, label_idx in zip(target['boxes'], target['labels']):
        xmin, ymin, xmax, ymax = box.tolist()
        width = xmax - xmin
        height = ymax - ymin
        
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, VOC_CLASSES[label_idx], color='white', backgroundcolor='red')
        
    plt.show()

if __name__ == "__main__":
    visualize_batch()