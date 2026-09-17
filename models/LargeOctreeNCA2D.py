
import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LargeNCA2D import LargeNCA2D
from models.OctreeNCA2D import OctreeNCA2D
from models.cholec80 import get_cholec_model

class LargeOctreeNCA2D(nn.Module):
    def __init__(self, octree_nca: OctreeNCA2D, num_channels: int, num_new_classes: int,
                 hidden_size: int):
        super(LargeOctreeNCA2D, self).__init__()
        self.do_ds = octree_nca.do_ds
        self.scale_factors = octree_nca.scale_factors
        self.upscale_factors = octree_nca.upscale_factors

        self.return_all_logits = False

        self.octree_nca = octree_nca

        new_backones = []
        for backbone_nca in self.octree_nca.backbone_ncas:
            new_backones.append(LargeNCA2D(backbone_nca, num_channels, num_new_classes, hidden_size))

        self.octree_nca.backbone_ncas = nn.ModuleList(new_backones)
        self.per_task_num_classes = [self.octree_nca.backbone_ncas[0].base_nca.num_classes,
                            num_new_classes]
        self.num_classes = sum(self.per_task_num_classes)
        self.num_channels = [self.octree_nca.backbone_ncas[0].base_nca.num_channels,
                                num_channels]


    def add_new_nca(self, num_channels: int, num_new_classes: int,
                 hidden_size: int):
        for i in range(len(self.octree_nca.backbone_ncas)):
            self.octree_nca.backbone_ncas[i].merge_and_init_new_nca(num_channels, 
                                                                    num_new_classes, 
                                                                    hidden_size)
        self.per_task_num_classes.append(num_new_classes)
        self.num_classes = sum(self.per_task_num_classes)
        self.num_channels.append(num_channels)

    def __downscale(self, x, level: int):
        if level==0:
            return x
        return F.interpolate(x, scale_factor=tuple(1/self.scale_factors[level]), mode="bilinear")

    def backbone_forward(self, x: torch.Tensor, return_states_all_levels:bool=False) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        x_downscaled = self.__downscale(x, len(self.octree_nca.backbone_ncas)-1)
        states: list[torch.Tensor] = self.octree_nca.backbone_ncas[-1].make_state(x_downscaled)

        seg_outputs = []
        states_all_levels: list[torch.Tensor] = []

        for level in list(range(len(self.octree_nca.backbone_ncas)))[::-1]: #micro to macro (low res to high res)

            states: list[torch.Tensor] = self.octree_nca.backbone_ncas[level].forward_internal(states)

            # get output channels
            new_logits = states[1][:,:self.per_task_num_classes[-1]]
            seg_outputs.append(new_logits)

            if return_states_all_levels:
                states_all_levels.append(torch.cat(states, dim=1)) # concat here will also clone, so we are good to go!

            if level > 0:
                # upsacele states to the next level
                x_downscaled = self.__downscale(x, level-1)
                # remove input channels
                states[0] = states[0][:, self.octree_nca.backbone_ncas[0].base_nca.num_input_channels:]
                states[0] = F.interpolate(states[0], scale_factor=2, mode='nearest')
                states[1] = F.interpolate(states[1], scale_factor=2, mode='nearest')
    

                states[0] = torch.cat([x_downscaled, states[0]], dim=1)
                
        if return_states_all_levels:
            return seg_outputs, states, states_all_levels
        return seg_outputs, states

    def compute_features(self, x: torch.Tensor, return_states_all_levels: bool) -> dict:
        if return_states_all_levels:
            seg_outputs, states, all_states = self.backbone_forward(x, return_states_all_levels=return_states_all_levels)
            H, W = x.shape[2], x.shape[3]
            all_states = [F.interpolate(s, size=(H,W), mode='nearest') for s in all_states]
        else:
            seg_outputs, states = self.backbone_forward(x)
            all_states = states
        assert not self.do_ds, "Cannot return all logits when doing deep supervision"
        return {
            'logits': self.get_all_logits_and_reorder(states),
            'features':  torch.cat(all_states, dim=1)
        }



    def forward(self, x: torch.Tensor):
        seg_outputs, states = self.backbone_forward(x)

        if self.return_all_logits:
            assert not self.do_ds, "Cannot return all logits when doing deep supervision"
            return self.get_all_logits_and_reorder(states)

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]
        
    def get_all_logits(self, states: list[torch.Tensor]) -> list[torch.Tensor]:
        # states that are from multiple merged NCAs
        assert len(states) == 2

        states_split = []
        start_channel = 0
        for c in (self.num_channels[:-1]): # the last channels are in states[1]
            states_split.append(states[0][:, start_channel:start_channel + c])
            start_channel += c
        states_split.append(states[1])
        # remove input channels
        states_split[0] = states_split[0][:, self.octree_nca.backbone_ncas[0].base_nca.num_input_channels:]

        #print("states_split", [s.shape for s in states_split])

        logits = []
        for c in self.per_task_num_classes:
            logits.append(states_split.pop(0)[:, :c])
        return logits

    def get_all_logits_and_reorder(self, states: list[torch.Tensor]) -> torch.Tensor:
        logits: list[torch.Tensor] = self.get_all_logits(states)
        
        return torch.cat(logits, dim=1)
        #print([l.shape for l in logits])
        logits = torch.cat([-1e4 * torch.ones(logits[0].shape[0], 1, logits[0].shape[2], logits[0].shape[3], device=logits[0].device), *logits, ], dim=1)
        

        order = [1,2,3,9,4,5,6,7,8,10,0,0,0,0,0]
        logits_reordered = logits[:, order]
        return logits_reordered
    
    def get_num_features(self, return_states_all_levels: bool = False) -> int:
        num_levels = len(self.octree_nca.backbone_ncas)
        if return_states_all_levels:
            return sum(self.num_channels) * num_levels
        return sum(self.num_channels) 
    
    def merge_ncas(self):
        for i in range(len(self.octree_nca.backbone_ncas)):
            self.octree_nca.backbone_ncas[i].merge_only()

    @staticmethod
    def load_from_file(folder: str, model_file: str, classes=None, num_channels=8, hidden_size=32, fire_rate=1.0) -> 'LargeOctreeNCA2D':
        if classes is None:
            with open(os.path.join(folder, 'classes.txt'), 'r') as f:
                lines = f.readlines()
            classes = [list(map(int, l.strip().split(','))) for l in lines]

        nca = get_cholec_model(len(classes[0]), fire_rate=fire_rate)

        nca = LargeOctreeNCA2D(nca, num_channels=num_channels, num_new_classes = len(classes[1]), hidden_size=hidden_size)
        nca.class_order = sum(classes, [])

        for i in range(2, len(classes)):
            nca.add_new_nca(num_channels=num_channels, num_new_classes=len(classes[i]), hidden_size=hidden_size)

        # Load on CPU first so checkpoints saved from CUDA can be exported on a
        # CPU-only workstation. The trainer moves the model to its target device.
        nca.load_state_dict(torch.load(
            os.path.join(folder, model_file),
            map_location="cpu",
            weights_only=True,
        ))
        print(f"Loaded LargeOctreeNCA2D from {folder} with classes {nca.class_order}")

        return nca
