import einops
from models.swin_unet.swin_unet_v2 import SwinTransformerSys
import torch
import torch.nn as nn
import torch.nn.functional as F


#https://github.com/chaineypung/Pytorch-Swin-Unet-V2/tree/main
class SurgicalSwinUNet(SwinTransformerSys):
    NUM_FEATURES = 96

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)


        return x

    def get_num_features(self) -> int:
        return self.NUM_FEATURES

    def compute_features(self, x: torch.Tensor) -> dict:
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        features = einops.rearrange(x, "B (H W) C -> B C H W ", H=64, W=64)
        logits = self.up_x4(x)
        features = F.interpolate(features, size=(256, 256), mode="bilinear", align_corners=False)
        return {
            'logits': logits,
            'features': features,
        }