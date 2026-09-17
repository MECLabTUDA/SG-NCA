from unet import UNet2D
import torch

class SurgicalUNet(UNet2D):
    NUM_FEATURES = 64
    
    def get_num_features(self) -> int:
        return self.NUM_FEATURES

    def compute_features(self, x: torch.Tensor) -> dict:
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        features = self.decoder(skip_connections, encoding)
        if self.monte_carlo_layer is not None:
            features = self.monte_carlo_layer(features)
        logits = self.classifier(features)
        return {
            'logits': logits,
            'features': features,
        }