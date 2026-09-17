import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput



class SurgicalMask2Former(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()

        model: Mask2FormerForUniversalSegmentation = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-coco-panoptic",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        self.model = model
        self.num_labels = num_labels


    
    def get_num_features(self) -> int:
        return self.NUM_FEATURES

    def forward_training(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        mask_labels = []
        class_labels = []
        for gt_semantic in masks:
        # gt_semantic: (H, W) with class ids
            instance_ids = torch.unique(gt_semantic)
            instance_ids = instance_ids[instance_ids != 0]  # ignore background

            mask_list = []
            label_list = []

            for cid in instance_ids:
                mask = (gt_semantic == cid)
                mask_list.append(mask)
                label_list.append(cid)

            mask_labels.append(torch.stack(mask_list).float())     # list length B
            class_labels.append(torch.tensor(label_list, device=gt_semantic.device))  # list length B


        outputs: Mask2FormerForUniversalSegmentationOutput = self.model(
            pixel_values=x,
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        return outputs.loss
    
    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        outputs: Mask2FormerForUniversalSegmentationOutput = self.model(
            pixel_values=x
        )
        class_probs = outputs.class_queries_logits.softmax(dim=-1)
        class_probs = class_probs[..., :-1]  # drop "no object"
        # (B, Q, C)

        mask_probs = outputs.masks_queries_logits.sigmoid()
        mask_probs = F.interpolate(
            mask_probs,
            size=x.shape[-2:],   # (H, W)
            mode="bilinear",
            align_corners=False
        )
        # (B, Q, H, W)

        # einsum is the cleanest way
        semantic_logits = torch.einsum(
            "bqc,bqhw->bchw",
            class_probs,
            mask_probs
        )

        return semantic_logits
    
    
    def compute_features(self, x: torch.Tensor) -> dict:
        raise NotImplementedError("Not implemented for Mask2Former")
        #transformer_decoder_last_hidden_state = self.model.mask2former.transformer_decoder.last_hidden_state

    def forward(self, x: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        if masks is not None:
            return self.forward_training(x, masks)
        else:
            return self.forward_inference(x)
