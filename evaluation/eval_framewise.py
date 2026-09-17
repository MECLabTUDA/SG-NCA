
import torch
import torch.nn as nn
import tqdm

from evaluation.dice import compute_dice, compute_dice_softmax

@torch.no_grad()
def eval_framewise(model: nn.Module, 
                   data_loader: torch.utils.data.DataLoader,
                   softmax: bool=False) -> torch.Tensor:
    
    model.eval()
    dices = []
    for i, batch in enumerate(tqdm.tqdm(data_loader, desc="Evaluating")):
        imgs, labels = batch
        imgs = imgs.to(next(model.parameters()).device)
        labels = labels.to(next(model.parameters()).device)

        outputs = model(imgs)

        if softmax:
            dices.append(compute_dice_softmax(outputs, labels, differentiable=False, ignore_index=data_loader.dataset.IGNORE_INDEX))
        else:
            dices.append(compute_dice(outputs, labels, differentiable=False))
        
    dices = torch.cat(dices, dim=0)
    return dices

