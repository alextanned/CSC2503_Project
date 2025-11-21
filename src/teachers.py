import torch
import torch.nn as nn
import clip
from torchvision import transforms as T

# Pascal VOC Classes (Same as dataset)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class TeacherManager(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print(f"Loading Teachers on {device}...")

        # --- 1. Load CLIP (Teacher for Logits/Classification) ---
        # ViT-B/32 is a good balance of speed/performance
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        # Pre-compute Text Embeddings for VOC Classes
        # This saves massive compute during training
        print("Building CLIP Text Prompts...")
        self.text_features = self._build_text_embeddings()

        # --- 2. Load DINOv2 (Teacher for Features/Localization) ---
        # 'dinov2_vits14' is the "Small" model. 
        # Use 'dinov2_vitb14' (Base) if you have >12GB VRAM.
        print("Loading DINOv2...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        self.dino_model.eval()
        
        # Standard DINO normalization (ImageNet defaults)
        self.dino_preprocess = T.Compose([
            T.Resize((224, 224)), # DINO expects multiples of 14, 224 is standard
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Freeze everything (We never update teachers)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False

    def _build_text_embeddings(self):
        """
        Generates text embeddings for 'a photo of a {class}'
        """
        prompts = [f"a photo of a {c}" for c in VOC_CLASSES]
        text_inputs = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            # Normalize for cosine similarity
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def get_clip_logits(self, images):
        """
        Returns: (Batch, 20) logits representing similarity to VOC classes
        """
        # CLIP expects specific resize/normalization. 
        # In a real loop, we might do this in the Dataset, but doing it here is safer for now.
        # We assume 'images' are standard floats [0,1]. CLIP needs its own preprocessing.
        # Note: This acts on the raw image tensors.
        
        # CLIP Forward Pass
        image_features = self.clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity: Image_Features @ Text_Features.T
        # Shape: (Batch, 512) @ (20, 512).T -> (Batch, 20)
        logits = 100.0 * image_features @ self.text_features.T
        return logits, image_features

    def get_dino_features(self, images):
        """
        Returns: Feature maps from DINO.
        Output Shape: (Batch, 384, H/14, W/14) for ViT-Small
        """
        # DINO Forward Pass
        # 'forward_features' gives dict with 'x_norm_clstoken', 'x_norm_patchtokens', etc.
        features_dict = self.dino_model.forward_features(images)
        
        # We want the patch tokens (spatial features)
        # Shape: (Batch, Num_Patches, Embed_Dim)
        patch_tokens = features_dict['x_norm_patchtokens']
        
        B, N, C = patch_tokens.shape
        # Reshape back to spatial grid. 
        # For 224x224 image, N=256 (16x16 patches).
        # We assume square images for simplicity here, or calculate based on input.
        H_grid = W_grid = int(N**0.5)
        
        spatial_features = patch_tokens.permute(0, 2, 1).reshape(B, C, H_grid, W_grid)
        return spatial_features

    def forward(self, images):
        """
        Main entry point. Pass the batch of images, get all teacher signals.
        """
        # 1. Resize/Norm for Teachers (Ideally done in dataloader, but doing here for safety)
        # We assume images are already normalized roughly, but let's apply DINO specific transform
        # Note: In production, optimization required here to avoid double-resizing.
        
        # DINO usually works best on standard ImageNet normalization
        dino_input = self.dino_preprocess(images)
        
        # CLIP Input (Needs separate resizing often 224 or 336)
        # We'll reuse the dino input for now as they are similar sizes/stats
        clip_input = dino_input 

        with torch.no_grad():
            clip_logits, clip_emb = self.get_clip_logits(clip_input)
            dino_features = self.get_dino_features(dino_input)
            
        return {
            "clip_logits": clip_logits,   # For Logit Distillation
            "clip_emb": clip_emb,         # Optional: Global embedding
            "dino_features": dino_features # For Feature Distillation
        }