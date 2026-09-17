
import torch


class SegWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    
class RelationWrapper(torch.nn.Module):
    def __init__(self, rel):
        super().__init__()
        self.rel = rel

    def forward(self, obj_feats, pair_idx):
        # obj_feats: [N, T, F]
        # pair_idx: [P, 2]
        return self.rel([obj_feats], [pair_idx])