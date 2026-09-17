
import torch
from data.cholec_scene_graph import CholecSceneGraphDataset
from PIL import Image
import data.cholec_vis as cholec_vis
import numpy as np


class SegmentationVisualizer:
    def __init__(self):
        pass

    def visualize_segmentation(self, trainer, frame_logits) -> Image.Image:
        assert frame_logits.dim() == 4  # (B, C, H, W)
        assert frame_logits.shape[0] == 1  # B=1 for visualization
        
        
        segmentation = trainer.get_index_map(frame_logits[0])  # (H, W)
        
        img = Image.fromarray((segmentation.cpu().numpy()).astype('uint8'), mode='L')

        img.putpalette(np.uint8(cholec_vis.get_cholecseg8k_colormap()))
        return img