import torch
import torch.nn as nn
import torch.nn.functional as F

from models.NCA2D import NCA2D

class NeuroSymbolicCA(nn.Module):
    def __init__(self, state_dim=16, num_classes=4, num_relations=2):
        super().__init__()
        self.nca = NCA2D(num_channels=state_dim, num_input_channels=3, num_classes=0, hidden_size=32,
                         fire_rate=1.0,
                         num_steps=64, use_norm=False)
        
        self.state_dim = state_dim
        
        # Segmentation Head (Pixel-wise)
        self.seg_head = nn.Conv2d(state_dim, num_classes, kernel_size=1)
        
        # Relation Head (Symbolic grounding)
        # Takes concatenated features of two objects
        self.rel_head = nn.Sequential(
            nn.Linear(state_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_relations) # 0: No Edge, 1: Edge
        )

    def forward(self, x, steps=60):
        # x shape: (Batch, 3, 64, 64) -> Initial RGB seed
        # Expand RGB to state_dim

        state = self.nca.make_state(x)
        
        state = self.nca.forward_internal(state)
            
        # 1. Neural Segmentation
        seg_logits = self.seg_head(state)
        
        return seg_logits, state

    def predict_relations(self, state, mask):
        """
        Extracts node features via Global Average Pooling on mask regions 
        and predicts relations between them.
        """
        batch_size = state.shape[0]
        # We assume 3 nodes: Circle(1), Triangle(2), Rectangle(3)
        node_indices = [1, 2, 3]
        relations = []

        for b in range(batch_size):
            node_feats = []
            for idx in node_indices:
                # Mask pixels belonging to this class
                m = (mask[b] == idx).float().unsqueeze(0) # (1, H, W)
                if m.sum() > 0:
                    feat = (state[b] * m).sum(dim=(1,2)) / m.sum()
                else:
                    feat = torch.zeros(self.state_dim, device=state.device)
                node_feats.append(feat)
            
            # Predict edges (0->1, 0->2, 1->2)
            # We specifically supervise 0->1 and 0->2 based on your dataset
            e01 = self.rel_head(torch.cat([node_feats[0], node_feats[1]]))
            e02 = self.rel_head(torch.cat([node_feats[0], node_feats[2]]))
            e12 = self.rel_head(torch.cat([node_feats[1], node_feats[2]]))
            relations.append(torch.stack([e01, e02, e12]))
            
        return torch.stack(relations)