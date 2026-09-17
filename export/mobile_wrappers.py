"""Fixed-shape wrappers used by the Android ONNX export.

The training pipeline builds object proposals dynamically in Python. Mobile
inference instead uses the fixed Cholec class and pair vocabularies, which
keeps both exported graphs small and gives Android a stable tensor contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MobileFrameWrapper(nn.Module):
    """Return class-ordered logits and compact per-object frame features."""

    def __init__(
        self,
        network: nn.Module,
        channel_order: Sequence[int],
        class_ids: Sequence[int],
        image_size: int = 256,
        mask_threshold: float = 0.5,
        area_threshold: int = 150,
    ) -> None:
        super().__init__()
        self.network = network
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.area_threshold = area_threshold

        reorder = [channel_order.index(class_id) for class_id in class_ids]
        self.register_buffer("channel_reorder", torch.tensor(reorder, dtype=torch.long))

        x_coordinates = torch.arange(image_size, dtype=torch.float32).reshape(1, 1, 1, image_size)
        y_coordinates = torch.arange(image_size, dtype=torch.float32).reshape(1, 1, image_size, 1)
        self.register_buffer("x_coordinates", x_coordinates)
        self.register_buffer("y_coordinates", y_coordinates)

    def forward(self, image: torch.Tensor):
        network_output = self.network.compute_features(
            image,
            return_states_all_levels=True,
        )
        segmentation_logits = torch.index_select(
            network_output["logits"],
            1,
            self.channel_reorder,
        )
        feature_map = network_output["features"]

        masks = (torch.sigmoid(segmentation_logits) > self.mask_threshold).to(feature_map.dtype)
        batch_size, num_classes, height, width = masks.shape
        pixel_counts = masks.sum(dim=(2, 3))

        flattened_masks = masks.reshape(batch_size, num_classes, height * width)
        flattened_features = feature_map.reshape(
            batch_size,
            feature_map.shape[1],
            height * width,
        ).transpose(1, 2)
        pooled_features = torch.bmm(flattened_masks, flattened_features)
        pooled_features = pooled_features / pixel_counts.clamp_min(1).unsqueeze(-1)

        normalized_area = (pixel_counts / float(self.image_size * self.image_size)).unsqueeze(-1)
        boolean_masks = masks > 0
        x_min = torch.where(
            boolean_masks,
            self.x_coordinates,
            torch.full_like(self.x_coordinates, float(self.image_size)),
        ).amin(dim=(2, 3))
        x_max = torch.where(
            boolean_masks,
            self.x_coordinates,
            torch.zeros_like(self.x_coordinates),
        ).amax(dim=(2, 3))
        y_min = torch.where(
            boolean_masks,
            self.y_coordinates,
            torch.full_like(self.y_coordinates, float(self.image_size)),
        ).amin(dim=(2, 3))
        y_max = torch.where(
            boolean_masks,
            self.y_coordinates,
            torch.zeros_like(self.y_coordinates),
        ).amax(dim=(2, 3))

        has_pixels = pixel_counts > 0
        zero = torch.zeros_like(x_min)
        bounding_boxes = torch.stack(
            [
                torch.where(has_pixels, x_min / self.image_size, zero),
                torch.where(has_pixels, y_min / self.image_size, zero),
                torch.where(has_pixels, x_max / self.image_size, zero),
                torch.where(has_pixels, y_max / self.image_size, zero),
            ],
            dim=-1,
        )

        object_features = torch.cat(
            [pooled_features, normalized_area, bounding_boxes],
            dim=-1,
        )
        presence = pixel_counts > self.area_threshold
        return segmentation_logits, object_features, presence


class MobileRelationWrapper(nn.Module):
    """Run the trained relation head for all fixed Cholec pairs."""

    def __init__(
        self,
        relation_classifier: nn.Module,
        class_ids: Sequence[int],
        possible_pairs: Sequence[tuple[int, int]],
    ) -> None:
        super().__init__()
        self.relation_classifier = relation_classifier
        class_slots = {class_id: index for index, class_id in enumerate(class_ids)}
        pair_slots = [
            (class_slots[subject], class_slots[object_])
            for subject, object_ in possible_pairs
        ]
        self.register_buffer("pair_slots", torch.tensor(pair_slots, dtype=torch.long))

    def forward(self, object_context: torch.Tensor):
        encoded_frames = self.relation_classifier.frame_encoder(object_context)
        encoded_clip = encoded_frames.reshape(encoded_frames.shape[0], -1)
        encoded_clip = self.relation_classifier.clip_encoder(encoded_clip)

        subject_features = torch.index_select(encoded_clip, 0, self.pair_slots[:, 0])
        object_features = torch.index_select(encoded_clip, 0, self.pair_slots[:, 1])
        pair_features = torch.cat([subject_features, object_features], dim=1)
        return self.relation_classifier.relation_predictor(pair_features)
