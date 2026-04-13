import torch
import torch.nn as nn

#todo reductions should be none or mean?

mse_loss = nn.MSELoss(reduction="mean")

def compute_domain_loss(
    domain_logits: torch.Tensor, 
    domain_labels: torch.Tensor, 
    class_weights: torch.Tensor = None
) -> torch.Tensor:
    """Compute the Cross Entropy loss between the model domain predictions and 
    and the domain labels.
    """
    if class_weights is not None:
        cross_entropy = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    else:
        cross_entropy = nn.CrossEntropyLoss(reduction="none")
    return cross_entropy(domain_logits, domain_labels)

def compute_gain_loss(
    z_hat: torch.Tensor, 
    z: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    """Compute Mean Squared Error of the imputations in latent space,
    only on the observed entries.
    """
    return mse_loss(z_hat * mask, z * mask)

def compute_reconstruction_loss(
    x_recon: torch.Tensor, 
    x: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    "Compute Root Mean Squared Error"
    x_zero_filled = x.clone()
    x_zero_filled[torch.isnan(x_zero_filled)] = 0
    squared_error = (x_recon - x_zero_filled) ** 2
    mse = (squared_error * mask).sum() / mask.sum()
    return torch.sqrt(mse)

def compute_task_specific_loss(
    gain_loss,
    reconstruction_loss,
    alpha: float,
    beta: float
):
    return alpha * gain_loss + beta * reconstruction_loss

def compute_model_loss(
    gain_loss,
    reconstruction_loss,
    domain_classifier_loss,
    alpha: float,
    beta: float,
    gamma: float
):
    "Loss that the model will optimize"
    return alpha * gain_loss + beta * reconstruction_loss + gamma * domain_classifier_loss

def compute_imputation_validation(
    x_imputed: torch.Tensor, 
    x: torch.Tensor, 
    missing_mask: torch.Tensor
):
    """
    Compute the imputation quality through RMSE, on the artificial missing entries
    on the original protein space.
    """
    if missing_mask.sum() == 0: # without missing entries
        return 0
    error = x_imputed[missing_mask] - x[missing_mask]
    squared_error = error ** 2
    sum_error = squared_error.sum()
    mse = sum_error / missing_mask.sum()
    return torch.sqrt(mse).item()
