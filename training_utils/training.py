import tqdm, wandb
import torch
import torch.nn.functional as F
from data.sampler import BalancedSampler, BalancedBatchSampler
from evaluation.dice import compute_dice, compute_dice_softmax
from evaluation.eval_framewise import eval_framewise
import torch.nn as nn

from models.surgical_mask2former import SurgicalMask2Former

def train(max_num_epochs, network: nn.Module, classes, dataset_class, batches_per_epoch, initial_lr, dryrun, 
          train_w_softmax=True, best_dice_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    train_dataset = dataset_class("train", classes=classes, return_for_softmax=train_w_softmax, filter_cases = False)
    sampler = BalancedSampler(train_dataset, num_samples=batches_per_epoch * 4)
    if len(classes) < 4:
        batch_sampler = BalancedBatchSampler(sampler, batch_size=4, num_batches=batches_per_epoch, force_classes=[c-1 for c in classes])
    else:
        batch_sampler = BalancedBatchSampler(sampler, batch_size=4, num_batches=batches_per_epoch)
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                         num_workers=4,
                                         batch_sampler=batch_sampler)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad==True, network.parameters()), initial_lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_num_epochs, eta_min=1e-6)

    best_dice = float('-inf')

    for epoch in range(max_num_epochs):
        network.train()
        loading_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_num_epochs}")
        losses = []
        bce_losses = []
        dice_losses = []
        for i, (img, label) in enumerate(loading_bar):
            optimizer.zero_grad()

            img = img.to(device)
            label = label.to(device)
            if network.__class__ == SurgicalMask2Former:
                loss = network.forward_training(img, label)
                loss_bce = torch.tensor(0.0)  # placeholder, not used for Mask2Former
                loss_dice = torch.tensor(0.0)  # placeholder, not used for Mask2Former
            else:
                logits = network(img)
                if train_w_softmax:
                    loss_bce = F.cross_entropy(logits, label, ignore_index=dataset_class.IGNORE_INDEX)
                    loss_dice = 1 - compute_dice_softmax(logits, label, differentiable=True, ignore_index=0).mean()
                else:
                    loss_bce = F.binary_cross_entropy_with_logits(logits, label)
                    loss_dice = 1 - compute_dice(logits, label, differentiable=True).mean()
                loss = loss_bce + loss_dice
            loss.backward()
            optimizer.step()
            loading_bar.set_postfix(loss=loss.item())
            
            bce_losses.append(loss_bce.item())
            dice_losses.append(loss_dice.item())
            losses.append(loss.item())
            if i % 1000 == 999:
                wandb.log({
                    "train/loss": sum(losses)/len(losses),
                    "train/bce_loss": sum(bce_losses)/len(bce_losses),
                    "train/dice_loss": sum(dice_losses)/len(dice_losses),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }, step=epoch * len(dataloader) + i)
                losses.clear()
                bce_losses.clear()
                dice_losses.clear()
            if dryrun and i > 100:
                break
        loader_val = torch.utils.data.DataLoader(dataset_class("val", classes=classes, original_only=True, return_for_softmax=train_w_softmax), batch_size=4, shuffle=False, num_workers=4)
        dices = eval_framewise(network, loader_val, softmax=train_w_softmax)
        if train_w_softmax:
            dices = dices[:,1:]  # ignore background
        log_dict = {
            f"val/dice_mean": dices.nanmean().item()
        }
        
        if best_dice_callback is not None:
            current_dice = dices.nanmean().item()
            if current_dice > best_dice:
                best_dice = current_dice
                best_dice_callback(best_dice, epoch)

        if classes is None:
            for i in range(dices.shape[1]):
                class_name = dataset_class.SCENE_GRAPH_DATASET.SEG_LABELS.inverse[i]
                log_dict[f"val/dice_{i} ({class_name})"] = dices[:,i].nanmean().item()
        else:
            for i, c in enumerate(classes):
                class_name = dataset_class.SCENE_GRAPH_DATASET.SEG_LABELS.inverse[c]
                log_dict[f"val/dice_{c} ({class_name})"] = dices[:,i].nanmean().item()

        wandb.log(log_dict, step=(epoch+1) * len(dataloader))
        lr_scheduler.step()