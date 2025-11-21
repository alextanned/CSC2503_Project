from torch.utils.tensorboard import SummaryWriter
import os

class ExperimentLogger:
    def __init__(self, log_dir="runs/distill_experiment"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Logging initialized at: {log_dir}")

    def log_scalars(self, metrics_dict, step):
        """
        Log a dictionary of scalars (e.g., {'loss/det': 0.5, 'loss/distill': 0.1})
        """
        for k, v in metrics_dict.items():
            self.writer.add_scalar(k, v, step)

    def log_images(self, tag, image_tensor, step):
        """
        Log batch of images
        """
        self.writer.add_images(tag, image_tensor, step)
        
    def close(self):
        self.writer.close()