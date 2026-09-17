
import torch
from models.LargeOctreeNCA2D import LargeOctreeNCA2D
import os
from models.cholec80 import get_cholec_model

def load_continual_nca(path: str):
    raise NotImplementedError("Loading continual NCA is not implemented yet")
    with open(os.path.join(path, 'classes.txt'), 'r') as f:
        lines = f.readlines()
    classes = [list(map(int, l.strip().split(','))) for l in lines]

    classes_stacked = sum(classes, [])
    if len(classes) > 1:
        return LargeOctreeNCA2D.load_from_file(path), classes_stacked
    
    nca = get_cholec_model(len(classes[0]))
    if len(classes) == 1:
        nca.load_state_dict(torch.load(os.path.join(path, "model.pth"), 
                                      map_location='cpu', weights_only=True))
        return nca, classes_stacked
    
    raise NotImplementedError("Loading continual NCA with two tasks is not implemented yet")