import torch
from training.loss import ajepa_loss

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch using masked A-JEPA loss.

    Args:
        model: AJEPA model
        dataloader: DataLoader for RealAudioDataset
        optimizer: torch optimizer
        device: torch.device ('cuda' or 'cpu')

    Returns:
        Average loss over the epoch
    """
    model.train()
    total_loss = 0

    for x_context, x_target in dataloader:
        # Move tensors to device
        x_context = x_context.to(device)  # [batch, frames, embed_dim]
        x_target = x_target.to(device)    # [batch, frames, embed_dim]

        # Create mask for masked positions (True where prediction is required)
        mask = (x_target != 0).any(dim=-1)  # [batch, frames]

        optimizer.zero_grad()
        z_pred, z_target = model(x_context, x_target)  # Forward pass

        # Compute masked MSE loss
        loss = ajepa_loss(z_pred, z_target, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

