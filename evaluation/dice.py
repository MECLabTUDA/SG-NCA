
import torch
import torch.nn.functional as F

def compute_dice_prob(pred_probs: torch.Tensor, 
                 target: torch.Tensor, 
                 epsilon: float = 1e-6,
                 differentiable: bool = False) -> torch.Tensor:
    # Flatten the tensors
    pred_flat = pred_probs.flatten(2)
    target_flat = target.flatten(2)

    # Compute intersection and sums
    intersection = (pred_flat * target_flat).sum(dim=2)
    pred_sum = pred_flat.sum(dim=2)
    target_sum = target_flat.sum(dim=2)

    # Compute Dice coefficient
    dice = (2.0 * intersection + epsilon) / (pred_sum + target_sum + epsilon)

    if not differentiable:
        contains_seg = target.sum(dim=[2,3]) > 0
        dice[~contains_seg] = torch.nan


    return dice

def compute_dice_softmax(logits: torch.Tensor, 
                 target_idx: torch.Tensor, 
                 epsilon: float = 1e-6,
                 differentiable: bool = False,
                 ignore_index: int = 255) -> torch.Tensor:
    assert logits.dim() == 4, "Logits must be (N, C, H, W)"
    assert target_idx.dim() == 3, "Target must be (N, H, W)"

    # (N, H, W) boolean mask
    valid_mask = target_idx != ignore_index

    if differentiable:
        pred_probs = torch.softmax(logits, dim=1)
    else:
        pred_probs = F.one_hot(
            torch.argmax(logits, dim=1),
            num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2).float()

    target_safe = target_idx.clone()
    target_safe[~valid_mask] = 0
    target = F.one_hot(
        target_safe, num_classes=logits.shape[1]
    ).permute(0, 3, 1, 2).float()

    # Expand mask to (N, C, H, W)
    valid_mask = valid_mask.unsqueeze(1)

    pred_probs = pred_probs * valid_mask
    target = target * valid_mask

    dice = compute_dice_prob(
            pred_probs, target, epsilon, differentiable
        )
    
    return dice



def compute_dice(logits: torch.Tensor, 
                 target: torch.Tensor, 
                 epsilon: float = 1e-6,
                 differentiable: bool = False) -> torch.Tensor:
    """
    Compute the Dice coefficient between predicted and target tensors.

    Args:
        pred (torch.Tensor): Predicted binary tensor of shape (N, H, W) or (N, C, H, W).
        target (torch.Tensor): Ground truth binary tensor of the same shape as pred.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    assert logits.shape == target.shape, f"Predicted and target tensors must have the same shape. Got {logits.shape} and {target.shape}."

    if differentiable:
        pred_probs = torch.sigmoid(logits)
    else:
        pred_probs = (logits > 0).float()

    return compute_dice_prob(pred_probs, target, epsilon, differentiable)