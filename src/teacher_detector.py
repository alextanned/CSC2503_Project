"""
Trainable detector that uses DINO and/or CLIP as frozen backbones with trainable detection heads.
This allows us to train a detector using teacher models and evaluate them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import clip


class TeacherBackboneWrapper(nn.Module):
    """Wraps DINO or CLIP to act as a FasterRCNN backbone."""
    
    def __init__(self, teacher_model, model_type='dino', freeze_backbone=True):
        super().__init__()
        self.teacher = teacher_model
        self.model_type = model_type
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.teacher.parameters():
                param.requires_grad = False
        
        # Set output channels based on model type
        if model_type == 'dino':
            self.out_channels = 384  # DINOv2 ViT-S
        elif model_type == 'clip':
            self.out_channels = 512  # CLIP ViT-B/32
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x):
        """
        Extract features from teacher model.
        Returns OrderedDict with single feature map for FasterRCNN.
        """
        B, C, H, W = x.shape
        
        if self.model_type == 'dino':
            # DINO expects 224x224
            if (H, W) != (224, 224):
                x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            else:
                x_resized = x
            
            # Get patch tokens
            features_dict = self.teacher.forward_features(x_resized)
            patch_tokens = features_dict['x_norm_patchtokens']  # (B, N, 384)
            
            B, N, C = patch_tokens.shape
            H_grid = W_grid = int(N ** 0.5)
            
            # Reshape to spatial: (B, 384, 16, 16) for 224x224 input
            features = patch_tokens.permute(0, 2, 1).reshape(B, C, H_grid, W_grid)
            
        elif self.model_type == 'clip':
            # CLIP visual encoder
            if (H, W) != (224, 224):
                x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            else:
                x_resized = x
            
            # Extract features from CLIP vision transformer (VisionTransformer forward)
            # The visual encoder expects input in the right format
            features = self.teacher(x_resized)  # (B, 512)
            
            # CLIP outputs (B, 512) by default, need to reshape to spatial
            # We'll use a simple broadcast to create a feature map
            features = features.unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
            features = features.expand(B, 512, 7, 7)  # Expand to 7x7 feature map
        
        # Return as OrderedDict for FasterRCNN
        return OrderedDict([('0', features)])


class TeacherDetector(nn.Module):
    """
    Detector using DINO or CLIP as frozen backbone with trainable Faster R-CNN head.
    """
    
    def __init__(
        self,
        teacher_type='dino',  # 'dino', 'clip', or 'both'
        num_classes=20,
        freeze_backbone=True,
        input_size=480,
        device='cuda'
    ):
        super().__init__()
        self.teacher_type = teacher_type
        self.input_size = input_size
        self.device = device
        
        print(f"\n[TeacherDetector] Building detector with {teacher_type} backbone...")
        
        # Load teacher models
        if teacher_type == 'dino':
            print("Loading DINOv2...")
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            backbone = TeacherBackboneWrapper(dino_model, 'dino', freeze_backbone)
            
        elif teacher_type == 'clip':
            print("Loading CLIP...")
            clip_model, _ = clip.load("ViT-B/32", device=device)
            # Convert CLIP to float32 to match input precision
            clip_model.visual.float()
            backbone = TeacherBackboneWrapper(clip_model.visual, 'clip', freeze_backbone)
        else:
            raise ValueError(f"Unknown teacher_type: {teacher_type}")
        
        self.backbone = backbone
        
        # Anchor generator for RPN
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # ROI pooling
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Build Faster R-CNN
        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes + 1,  # +1 for background
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=input_size,
            max_size=input_size,
        )
        
        print(f"[TeacherDetector] Created with {backbone.out_channels}-dim features")
    
    def forward(self, images, targets=None):
        """
        Forward pass compatible with training and evaluation.
        
        Training mode (targets provided):
            Returns: (loss_dict, None, None)
        
        Eval mode (targets=None):
            Returns: (detections, None, None)
        """
        # Convert to list if tensor
        if isinstance(images, torch.Tensor):
            image_list = list(images)
        else:
            image_list = images
        
        if self.training and targets is not None:
            # Training mode: return losses
            loss_dict = self.detector(image_list, targets)
            return loss_dict, None, None
        else:
            # Eval mode: return detections
            detections = self.detector(image_list)
            return detections, None, None
    
    def train(self, mode=True):
        """Override train to handle detector"""
        super().train(mode)
        self.detector.train(mode)
        return self
    
    def eval(self):
        """Override eval to handle detector"""
        super().eval()
        self.detector.eval()
        return self
