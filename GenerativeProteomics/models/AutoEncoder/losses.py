import torch
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