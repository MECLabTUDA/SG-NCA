
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import torch.nn as nn
import torch.nn.functional as F

class SurgicalSegFormer(nn.Module):
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
        out = self.model(x)
        pred = out.logits
        pred = F.interpolate(pred, size=x.shape[2:], mode='bilinear', align_corners=False)
        return pred
    

    def compute_features(self, x: torch.Tensor) -> dict:
        out = self.model(x, output_hidden_states=True)
        pred = out.logits
        pred = F.interpolate(pred, size=x.shape[2:], mode='bilinear', align_corners=False)

        states = out.hidden_states
        states = [F.interpolate(s, size=(256,256)) for s in states]
        states = torch.concat(states, dim=1)  # (B, C, H, W)
        
        return {
            'logits': pred,
            'features': states,
        }