import torch
import logging
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef

from models.GainPro.gain import Gain
from models.GainPro.metrics import Metrics
from utils.configs.training_config import GainTrainingConfig
from utils.data.helper import generate_hint
from utils.writers.experiment_writer import ExperimentWriter

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: Gain,
        training_hypers: GainTrainingConfig,
    ) -> "Trainer":
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model

        self.num_epochs = training_hypers.num_epochs

        self.generator_lr = training_hypers.generator_lr
        self.discriminator_lr = training_hypers.discriminator_lr
        self._init_optimizers()

        self.alpha = training_hypers.alpha
        self.hint_rate = training_hypers.hint_rate

        self.metrics = Metrics()

    def _init_optimizers(
        self,
    ) -> None:
        self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.generator_lr, weight_decay=0.5)
        self.optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.discriminator_lr, weight_decay=0.5)
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=100, gamma=0.9)

    def _step_schedulers(
        self
    ) -> None:
        min_lr = 1e-100
            
        current_lr_d = self.optimizer_D.param_groups[0]['lr']
        if current_lr_d > min_lr:
            self.scheduler_D.step()

    def generate_sample(
        self,
        x, 
        mask,
        Z,
    ) -> torch.Tensor:
        logger.debug("\n In generate sample")
        mask = mask.float()
        x_noise = mask * x + (1 - mask) * Z
        input_G = torch.cat((x_noise, mask), 1).float()
        logger.debug(
            f"\n Input G shape {input_G.shape}"
        )
        x_hat = self.model.generator(input_G)
        x_hat.to(torch.float32)
        logger.debug(
            f"\n Mask:\n {mask}"
            f"\n X:\n {x}"
            f"\n Z:\n {Z}"
            f"\n X with noise:\n {x_noise}"
        )
        return x_hat

    def _loss_discriminator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.Tensor:
        logger.debug(f"\n In loss discriminator")
        logger.debug(
            f"\n x shape: {x.shape}"
            f"\n mask shape: {mask.shape}"
            f"\n hint shape: {hint.shape}"
            f"\n Z shape: {Z.shape}"
        )

        mask = mask.float()
        sample_G = self.generate_sample(x=x, mask=mask, Z=Z)
        x_hat = mask * x + (1 - mask) * sample_G

        mask_hat = self.model.discriminator(x_hat.detach(), hint)
        logger.debug(
            f"\n Mask:\n {mask}"
            f"\n Sample G:\n {sample_G}"
            f"\n X hat:\n {x_hat}"
            f"\n Mask hat:\n {(torch.sigmoid(mask_hat) > 0.5).float()}" # 1 = observed, 0 = missing
        )

        # Only train on positions where hint == 0.5 (b_i = 0, i.e. unknown to discriminator)
        b_mask = torch.isclose(hint, torch.full_like(hint, 0.5))

        loss_D = (nn.BCEWithLogitsLoss(reduction="mean")(
            mask_hat[b_mask], 
            mask[b_mask]
        ))
        return loss_D

    def _loss_generator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.Tensor:
        mask = mask.float()
        sample_G = self.generate_sample(x=x, mask=mask, Z=Z)
        x_hat = mask * x + (1 - mask) * sample_G
        mask_hat = self.model.discriminator(x_hat, hint)

        # Only train on positions where hint == 0.5 (b_i = 0, i.e. unknown to discriminator)
        target = torch.ones_like(mask_hat, device=self.device).float()
        b_mask = (hint == 0.5)
        loss_G_entropy = (
            nn.BCEWithLogitsLoss(reduction="mean")(
                mask_hat[b_mask], 
                target[b_mask],
            )
        )

        observed_entries = mask.bool()
        loss_G_mse = (
            nn.MSELoss(reduction="mean")(
                sample_G[observed_entries].float(), 
                x[observed_entries].float(),
            )
        )
        loss_G = loss_G_entropy + self.alpha * loss_G_mse
        
        imputed_values = sample_G[mask == 0]
        reconstructed_values = sample_G[mask == 1]
        logger.debug(
            f"\n Loss Generator:"
            f"\n Generator output min/max: {sample_G.min().item(), sample_G.max().item()}"
            f"\n Generator output mean: {sample_G[mask==1].mean(dim=0).item()}"
            f"\n Generator output std: {sample_G.std(dim=0).mean().item()}"
            f"\n Reconstructed (Observed):"
            f"\n   Mean: {reconstructed_values.mean():.4f}"
            f"\n   Std: {reconstructed_values.std():.4f}"
            f"\n   Min: {reconstructed_values.min():.4f}"
            f"\n   Max: {reconstructed_values.max():.4f}"
            f"\n Imputed (Missing):"
            f"\n   Mean: {imputed_values.mean():.4f}"
            f"\n   Std: {imputed_values.std():.4f}"
            f"\n   Min: {imputed_values.min():.4f}"
            f"\n   Max: {imputed_values.max():.4f}"
            f"\n Generator MSE (observed): {loss_G_mse.item()}"
            f"\n Discriminator entropy: {loss_G_entropy.item()}"
        )
        return loss_G, loss_G_mse, loss_G_entropy

    def _update_discriminator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.Tensor:
        loss_D = self._loss_discriminator(
            x=x,
            mask=mask,
            hint=hint,
            Z=Z,
        )
        self.optimizer_D.zero_grad()
        loss_D.backward()
        d_norm = sum(p.grad.norm().item() ** 2
            for p in self.model.discriminator.parameters()
            if p.grad is not None) ** 0.5
        logger.debug(f"\n Discriminator's gradient norm: {d_norm:.4f}")
        torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
        self.optimizer_D.step()
        return loss_D

    def _update_generator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.Tensor:
        loss_G, generator_mse, generator_entropy = self._loss_generator(
            x=x,
            mask=mask,
            hint=hint,
            Z=Z,
        )
        self.optimizer_G.zero_grad()
        loss_G.backward()
        g_norm = sum(p.grad.norm().item() ** 2
            for p in self.model.generator.parameters()
            if p.grad is not None) ** 0.5
        logger.debug(f"\n Generator's gradient norm: {g_norm:.4f}")
        torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_norm=1.0)
        self.optimizer_G.step()
        return loss_G, generator_mse, generator_entropy

    def epoch(
        self,
        x,
        x_true,
        Z,
        mask,
        hint,
        train_mode: bool=True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if train_mode:
            self.model.train()
            discriminator_loss = self._update_discriminator(x=x, mask=mask, hint=hint, Z=Z)
            generator_loss, generator_mse, generator_entropy = self._update_generator(x=x, mask=mask, hint=hint, Z=Z)
        else:
            self.model.eval()
            with torch.no_grad():
                discriminator_loss = self._loss_discriminator(x=x, mask=mask, hint=hint, Z=Z)
                generator_loss, generator_mse, generator_entropy = self._loss_generator(x=x, mask=mask, hint=hint, Z=Z)

        x_hat = self.generate_sample(x=x, mask=mask, Z=Z)

        mse = nn.MSELoss()(x_true, x_hat)
        rmse = np.sqrt(mse.detach().cpu().numpy())

        if train_mode:
            self.model.train()

        return (
            discriminator_loss.detach().clone(), 
            generator_loss.detach().clone(), 
            generator_mse.detach().clone(), 
            generator_entropy.detach().clone(), 
            rmse
        )

    def train(
        self,
        x_train: torch.Tensor,
        x_true: torch.Tensor,
        observed_mask: torch.Tensor,
        experiment_writer: ExperimentWriter,
    ) -> None:
        for ep in range(1, self.num_epochs+1):
            logger.info(f"\t Epoch {ep}/{self.num_epochs}")

            hint = generate_hint(observed_mask.detach().cpu().numpy(), self.hint_rate)
            hint = hint.to(self.device)

            discriminator_loss, generator_loss, generator_rmse, generator_entropy, rmse = self.epoch(
                x=x_train,
                x_true=x_true,
                mask=observed_mask,
                hint=hint,
                Z=torch.randn(x_train.shape, device=self.device),
                train_mode=True
            )

            self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss.item())
            self.metrics.train_metrics["generator_loss"].append(generator_loss.item())
            self.metrics.train_metrics["generator_rmse"].append(generator_rmse.item())
            self.metrics.train_metrics["generator_entropy"].append(generator_entropy.item())
            self.metrics.train_metrics["rmse"].append(rmse)

            self._step_schedulers()

        experiment_writer.metrics_writer.log_metrics(metrics=self.metrics)

    def impute(
        self,
        x_missing,
        mask,
    ) -> np.ndarray:
        # Invert the mask because artificially missing entries are marked with 1s,
        # while `generate_sample` expects observed entries to be marked with 1s.
        mask = ~mask
        x_hat = self.generate_sample(
            x=x_missing, 
            mask=mask,
            Z=torch.randn(x_missing.shape, device=self.device),
        )
        return x_hat.detach().cpu().numpy()

    def discriminate(
        self,
        x,
        mask,
        mask_observed,
    ) -> torch.Tensor:
        """
        Args:
            - mask (torch.Tensor): Artificial missing entries (1 = artificial missing).
        """
        hint = generate_hint(mask.detach().cpu().numpy(), hint_rate=self.hint_rate)
        hint = torch.tensor(hint, device=self.device, dtype=torch.float32)
        sample_G = self.generate_sample(
            x=x, 
            mask=mask, 
            Z=torch.randn(x.shape, device=self.device),
        )
        logger.debug(
            f"\n Data mean: {x[mask_observed.bool()].mean().item()}"
            f"\n Generator output mean: {sample_G.mean().item()}"
            f"\n Generator output std: {sample_G.std(dim=0).mean().item()}"
        )

        # Input holdout entries given by the eval mask (`mask`)
        mask = (~mask).float()
        fake_X = mask * x + (1 - mask) * sample_G

        # Input original missing entries given by the observed mask (`mask`)
        mask_observed = mask_observed.float()
        fake_X = fake_X * mask_observed + sample_G * (1 - mask_observed)

        logits = self.model.discriminator(fake_X.detach(), hint)
        mask_hat = (torch.sigmoid(logits) > 0.5).float() # 1 = observed, 0 = missing

        probs = torch.sigmoid(logits)

        logger.debug(
            f"Hint:\n {hint}"
            f"Sample G:\n {sample_G}"
            f"Holdout entries mask:\n {mask}"
            f"Fake X:\n {fake_X}"
            f"Observed mask mean: {mask_observed.mean().item()}"
            f"Discriminator certainty:\n {torch.sigmoid(logits)}"
            f"  Mean: {probs.mean().item()}"
            f"  Std: {probs.std().item()}"
            f"Observed mask:\n {mask_observed}"
            f"Predicted mask:\n {mask_hat}"
        )
        return mask_hat

    def compute_precision_recall_discriminator(
        self,
        x,
        mask: torch.Tensor,
        mask_observed: torch.Tensor,
        positive_label: int,
    ) -> dict[str, float]:
        label_description = {1: "Observed Entry", 0: "Missing Entry"}.get(positive_label, "Unknown")
        logger.info(f"\n Positive label: {positive_label} ({label_description})")
        
        mask_hat = self.discriminate(
            x,
            mask,
            mask_observed,
        )
        y_true = mask_observed.detach().cpu().numpy().ravel()
        y_pred = mask_hat.detach().cpu().numpy().ravel()
        logger.debug(
            f"\n True positives: {((y_true == 1) & (y_pred == 1)).sum()}"
            f"\n False positives: {((y_true == 0) & (y_pred == 1)).sum()}"
            f"\n True negatives: {((y_true == 0) & (y_pred == 0)).sum()}"
            f"\n False negatives: {((y_true == 1) & (y_pred == 0)).sum()}"
        )

        precision = precision_score(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=positive_label,
            average="binary",
        )
        recall = recall_score(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=positive_label,
            average="binary",
        )
        matthews = matthews_corrcoef(
            y_true=y_true,
            y_pred=y_pred,
        )

        logger.info(
            f"\n Discriminator:"
            f"\n   Precision: {precision:.4f}"
            f"\n   Recall: {recall:.4f}"
            f"\n   Matthews Correlation Coefficient: {matthews:.4f}"
        )
        results = {"discriminator_precision": precision, "discriminator_recall": recall}
        return results