
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import torch.nn as nn
import torch.nn.functional as F

class SurgicalSegFormer2(nn.Module):
    NUM_FEATURES = 512
    def __init__(self, num_labels:int):
        super().__init__()
        config = SegformerConfig(num_channels=3)
        config.num_labels = num_labels
        model = SegformerForSemanticSegmentation(config)
        self.model = model

    def get_num_features(self) -> int:
        return self.NUM_FEATURES

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        out = self.model(x)
        pred = out.logits
        return pred
    

    def compute_features(self, x: torch.Tensor) -> dict:
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        out = self.model(x, output_hidden_states=True)
        pred = out.logits

        states = out.hidden_states
        states = [F.interpolate(s, size=(256,256)) for s in states]
        states = torch.concat(states, dim=1)  # (B, C, H, W)
        
        return {
            'logits': pred,
            'features': states,
        }