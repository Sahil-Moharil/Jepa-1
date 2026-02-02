import torch
import torch.nn.functional as F

def ajepa_loss(pred, target, mask):
    """
    Computes masked MSE loss for A-JEPA.
    pred: [batch, frames, embed_dim]
    target: [batch, frames, embed_dim]
    mask: [batch, frames] (bool tensor, True = masked)
    """
    # Expand mask to match embedding dimension
    mask = mask.unsqueeze(-1).expand_as(pred)
    masked_pred = pred[mask]
    masked_target = target[mask]

    return F.mse_loss(masked_pred, masked_target)

