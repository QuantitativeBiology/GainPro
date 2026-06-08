import torch
import torch.nn as nn
import torch.nn.functional as F

def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.float()
    mse = F.mse_loss(x_hat, x, reduction="none")
    mse_masked = mse * mask
    rmse = torch.sqrt(mse_masked.sum() / mask.sum())
    return rmse

def discriminator_mask_loss(
    mask: torch.Tensor,
    mask_hat: torch.Tensor,
    hint: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the binary cross-entropy loss for the GAIN discriminator 
    only on positions where the components are unknown to it (hint == 0.5).
    """
    # Only train on positions where hint == 0.5 (b_i = 0, i.e. unknown to discriminator)
    b_mask = torch.isclose(hint, torch.full_like(hint, 0.5))
    if b_mask.sum() == 0:
        return torch.tensor(0.0, device=mask.device, requires_grad=True)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    return criterion(mask_hat[b_mask], mask[b_mask])

def generator_mask_loss(
    mask: torch.Tensor,
    mask_hat: torch.Tensor,
    hint: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the binary cross-entropy loss for the GAIN discriminator 
    only on positions where the components are unknown to it (hint == 0.5).
    """
    # Only train on positions where hint == 0.5 (b_i = 0, i.e. unknown to discriminator)
    b_mask = torch.isclose(hint, torch.full_like(hint, 0.5))
    if b_mask.sum() == 0:
        return torch.tensor(0.0, device=mask.device, requires_grad=True)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    # Generator wants discriminator maximally confused (output = 0.5)
    targets = torch.full_like(mask_hat[b_mask], 0.5)
    return criterion(mask_hat[b_mask], targets)