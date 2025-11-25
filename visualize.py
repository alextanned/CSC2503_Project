import argparse
import os

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.dataset import get_loaders
from src.students import StudentDetector
from utils.utils import plot_prediction_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize predictions of a trained detector on VOC")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to student .pth checkpoint")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small", "vit_tiny"],
                        help="Backbone architecture used in the student model")

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for visualization")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., 'cuda', 'cpu'); default auto-detect")
    parser.add_argument("--input-size", type=int, default=480,
                        help="Input image size (assumed square)")
    parser.add_argument("--num-images", type=int, default=32,
                        help="Total number of images to visualize")
    parser.add_argument("--score-thresh", type=float, default=0.3,
                        help="Score threshold for showing predicted boxes")
    parser.add_argument("--output", type=str, default="viz_outputs",
                        help="Directory to save visualization grids")
    parser.add_argument("--images-per-grid", type=int, default=16,
                        help="How many images to pack into one grid")
    
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output, exist_ok=True)

    # 1. Data (use val loader only)
    _, val_loader = get_loaders(batch_size=args.batch_size, input_size=args.input_size)

    # 2. Model
    model = StudentDetector(
        model_type=args.backbone,
        num_classes=20,
        teacher_feature_dim=384,
        use_feature_distill=False,
        use_logit_distill=False,
        input_size=args.input_size,
    ).to(device)
    model.eval()

    # 3. Load checkpoint
    assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 4. Iterate over val set and collect visualizations
    images_to_show = args.num_images
    images_per_grid = args.images_per_grid
    grid_idx = 0

    pbar = tqdm(val_loader, desc="Visualizing")
    for batch in pbar:
        if images_to_show <= 0:
            break

        student_imgs, teacher_imgs, targets = batch
        student_imgs = student_imgs.to(device)

        outputs, _, _ = model(student_imgs)

        # How many from this batch do we actually want?
        batch_limit = min(len(student_imgs), images_to_show, images_per_grid)

        # Use your plotting util to overlay GT + predictions
        canvases = plot_prediction_batch(
            student_imgs,
            targets,
            predictions=outputs,
            limit=batch_limit,
            score_thresh=args.score_thresh,
        )  # (N, 3, H, W)

        # Create a grid to save
        grid = make_grid(canvases, nrow=min(batch_limit, int(images_per_grid**0.5)))
        out_path = os.path.join(args.output, f"viz_grid_{grid_idx:03d}.png")
        save_image(grid / 255.0, out_path)  # grid is uint8 [0,255] → [0,1] float

        print(f"Saved {out_path}")

        images_to_show -= batch_limit
        grid_idx += 1

    print("Visualization complete.")


if __name__ == "__main__":
    main()
