import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from collections import OrderedDict
import timm


class FeatureAdapter(nn.Module):
    """Maps student features to teacher dimension (e.g. 512 -> 384)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.adapter(x)


class ViTFeatureWrapper(nn.Module):
    """
    Wraps a ViT to output 2D feature maps (B, C, H, W) instead of a sequence.
    Assumes square inputs and patch-based ViT (e.g., vit_tiny_patch16_224).
    """
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        # timm ViTs usually expose this
        self.embed_dim = getattr(vit_model, "embed_dim", vit_model.num_features)

    def forward(self, x):
        # forward_features for timm ViT returns (B, N+1, C) or a dict; handle both
        out = self.vit.forward_features(x)
        if isinstance(out, dict):
            # common timm behavior: out["x"] or out["last"]
            if "x" in out:
                tokens = out["x"]
            elif "last" in out:
                tokens = out["last"]
            else:
                # fallback: first value in dict
                tokens = next(iter(out.values()))
        else:
            tokens = out

        # Remove CLS token at index 0 -> (B, N, C)
        tokens = tokens[:, 1:, :]
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)

        tokens = tokens.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        return tokens


class BackboneWithHook(nn.Module):
    """
    Wrap generic backbone so FasterRCNN can use it, and we can grab mid-level features
    for distillation.
    """
    def __init__(self, body: nn.Module, out_channels: int):
        super().__init__()
        self.body = body
        self.out_channels = out_channels
        self._feat_for_distill = None

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.body(x)
        # handle cases where body returns dict/list already
        if isinstance(feats, dict):
            # assume single-key dict
            f = next(iter(feats.values()))
        elif isinstance(feats, (list, tuple)):
            f = feats[-1]
        else:
            f = feats

        # store mid-level feature map for distillation
        self._feat_for_distill = f

        # FasterRCNN expects an OrderedDict of feature maps
        # single-scale backbone: just one entry named "0"
        return OrderedDict([("0", f)])

    @property
    def distill_features(self):
        return self._feat_for_distill


def build_backbone(model_type: str = "resnet18", pretrained: bool = True):
    """
    Returns (body_module, out_channels) for the chosen backbone.
    """
    model_type = model_type.lower()

    if model_type == "resnet18":
        base = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # strip avgpool + fc
        body = nn.Sequential(*list(base.children())[:-2])
        out_channels = 512

    elif model_type in ["mobilenet", "mobilenet_v3_small", "mobilenet-small"]:
        base = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        body = base.features  # feature extractor
        out_channels = 576  # last conv channels for v3-small

    elif model_type in ["vit_tiny", "vit_tiny_patch16_224"]:
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained)
        body = ViTFeatureWrapper(vit)
        out_channels = body.embed_dim

    else:
        raise ValueError(f"Unknown backbone model_type: {model_type}")

    return body, out_channels


class StudentDetector(nn.Module):
    """
    Student = (backbone + FasterRCNN head) + optional distillation taps.

    forward(images, targets=None) returns:
      - train:  loss_dict, student_features, student_logits
      - eval:   detections, student_features, student_logits
    """
    def __init__(
        self,
        model_type: str = "resnet18",
        num_classes: int = 20,
        teacher_feature_dim: int = 384,
        use_feature_distill: bool = True,
        use_logit_distill: bool = True,
    ):
        super().__init__()
        self.model_type = model_type
        print(f"[StudentDetector] Using backbone: {model_type}")

        # 1. backbone
        body, out_channels = build_backbone(model_type)
        self.backbone = BackboneWithHook(body, out_channels)

        # 2. FasterRCNN detection head
        # Single-scale anchor generator (we only have feature map "0")
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )

        # ROI pooling over feature map "0"
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2,
        )

        # +1 for background
        self.detector = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes + 1,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

        # 3. distillation taps
        self.use_feature_distill = use_feature_distill
        self.use_logit_distill = use_logit_distill

        if use_feature_distill:
            self.projector = FeatureAdapter(out_channels, teacher_feature_dim)
        else:
            self.projector = None

        if use_logit_distill:
            self.aux_classifier = nn.Linear(out_channels, num_classes)
        else:
            self.aux_classifier = None

    def _get_distill_features_and_logits(self):
        feat = self.backbone.distill_features
        if feat is None:
            return None, None

        student_features = None
        student_logits = None

        if self.projector is not None:
            student_features = self.projector(feat)

        if self.aux_classifier is not None:
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, C)
            student_logits = self.aux_classifier(pooled)

        return student_features, student_logits

    def forward(self, images: torch.Tensor, targets=None):
        """
        images: (B, C, H, W) tensor
        targets: list[dict] in torchvision detection format
        """
        # FasterRCNN expects list of (C,H,W) tensors
        if isinstance(images, torch.Tensor):
            image_list = list(images)
        else:
            image_list = images

        if self.training and targets is not None:
            # 1) detection loss dict from FasterRCNN
            loss_dict = self.detector(image_list, targets)  # dict of scalars

            # 2) distillation taps
            student_features, student_logits = self._get_distill_features_and_logits()
            return loss_dict, student_features, student_logits
        else:
            # 1) detections at inference
            detections = self.detector(image_list)
            # 2) distillation taps (optional)
            student_features, student_logits = self._get_distill_features_and_logits()
            return detections, student_features, student_logits
