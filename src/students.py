import torch
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDHead

class FeatureAdapter(nn.Module):
    """
    Maps Student Feature dimensions to Teacher Feature dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.adapter(x)

class StudentDetector(nn.Module):
    def __init__(self, model_type="mobilenet", num_classes=20, teacher_feature_dim=384):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "mobilenet":
            print("Initializing MobileNetV3-Large SSDLite...")
            self.detector = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
            
            self.student_feat_dim = 672 
            
            in_channels = [672, 480, 512, 256, 256, 128]
            
            num_anchors = self.detector.anchor_generator.num_anchors_per_location()
            
            # Rebuild the head with the correct input channels
            self.detector.head = SSDHead(in_channels, num_anchors, num_classes + 1)
            
        else:
            raise NotImplementedError("Only 'mobilenet' is supported for Phase 3.")

        self.intermediate_features = {}
        self._register_hooks()

        self.projector = FeatureAdapter(self.student_feat_dim, teacher_feature_dim)

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                # We want Key '0' (the high-res feature map)
                if isinstance(output, dict):
                    if '0' in output:
                        feature_map = output['0']
                    else:
                        # Fallback: grab the first available value
                        feature_map = list(output.values())[0]
                elif isinstance(output, (list, tuple)):
                    feature_map = output[0]
                else:
                    feature_map = output
                
                self.intermediate_features[name] = feature_map
            return hook

        if self.model_type == "mobilenet":
            self.detector.backbone.register_forward_hook(get_activation('mid_level'))

    def forward(self, images, targets=None):
        """
        Returns:
           - loss_dict (training) or detections (inference)
           - projected_features (for distillation)
        """
        self.intermediate_features = {}
        
        output = self.detector(images, targets)
        
        if 'mid_level' in self.intermediate_features:
            raw_feat = self.intermediate_features['mid_level']
            projected_feat = self.projector(raw_feat)
        else:
            print("WARNING: Hook failed to capture features.")
            projected_feat = None
        
        return output, projected_feat