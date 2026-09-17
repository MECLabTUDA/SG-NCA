from unet import UNet2D
import torch

class TinySurgicalUNet(UNet2D):
    NUM_FEATURES = 4

    def __init__(self, in_channels=3, out_classes=2):
        super().__init__(
            in_channels=in_channels,
            out_classes=out_classes,
            out_channels_first_layer=4,
            num_encoding_blocks=5,
            padding="same",
        )
    

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