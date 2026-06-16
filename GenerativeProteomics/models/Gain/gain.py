import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
    EarlyStoppingReason
)

from utils.data.helper import generate_hint
from utils.configs.model_config import GainConfig
from utils.configs.training_config import GainTrainingConfig
from models.GainPro.generator import Generator
from models.GainPro.discriminator import Discriminator
from models.GainPro.losses import (
    reconstruction_loss,
    discriminator_mask_loss,
    generator_mask_loss
)

logger = logging.getLogger(__name__)


class Gain(pl.LightningModule):
    """
    GAIN: Generative Adversarial Imputation Network.

    Dataloader contract (TensorDataset order):
        Training:   (x_true, x_missing, observed_mask, artificial_mask)
        Validation: (x_true, x_missing, observed_mask, artificial_mask)
        Inference:  (x_missing, observed_mask)

    Where:
        x_true          – ground-truth values; NaN-filled originals set to 0
        x_missing       – fill-strategy input (zeros or mean); missing entries filled
        observed_mask   – 1 where entry was originally observed, 0 where originally missing
        artificial_mask – 1 where an observed entry was deliberately hidden for supervision
                          (subset of observed_mask); 0 everywhere else

    The reconstruction loss is computed ONLY on artificial_mask positions,
    which are the only positions where we have clean ground-truth supervision.
    """

    def __init__(
        self,
        input_dim: int,
        hypers: GainConfig,
        training_cfg: GainTrainingConfig,
        generator_output_activation: nn.Module = None,
    ) -> None:
        super().__init__()

        self.automatic_optimization = False

        self.input_dim = input_dim
        hidden_dim = hypers.hidden_dim
        self.hidden_dim = int(self.input_dim / 2) if hidden_dim is None else hidden_dim
        logger.debug(
            f"\n Input dim: {self.input_dim}"
            f"\n Hidden dim: {self.hidden_dim}"
        )

        self.generator = Generator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=hypers.num_hidden_layers_generator,
            generator_output_activation=generator_output_activation,
        )
        self.discriminator = Discriminator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=hypers.num_hidden_layers_discriminator,
        )

        self.training_cfg = training_cfg

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        x_hat = self.generator(x=batch["x"], mask=batch["observed_mask"])
        x_imputed = batch["observed_mask"] * batch["x"] + (1 - batch["observed_mask"]) * x_hat
        mask_hat = self.discriminator(x_imputed=x_imputed, hint=batch["hint"])
        return {"x_hat": x_hat, "x_imputed": x_imputed, "mask_hat": mask_hat}

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.training_cfg.generator_optimizer.lr,
            weight_decay=self.training_cfg.generator_optimizer.weight_decay,
        )
        discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=self.training_cfg.discriminator_optimizer.lr,
            weight_decay=self.training_cfg.discriminator_optimizer.weight_decay,
        )
        optimizers = [generator_optimizer, discriminator_optimizer]

        schedulers = []
        if self.training_cfg.generator_scheduler is not None:
            schedulers.append(torch.optim.lr_scheduler.StepLR(
                generator_optimizer,
                step_size=self.training_cfg.generator_scheduler.step,
                gamma=self.training_cfg.generator_scheduler.gamma,
            ))
        if self.training_cfg.discriminator_scheduler is not None:
            schedulers.append(torch.optim.lr_scheduler.StepLR(
                discriminator_optimizer,
                step_size=self.training_cfg.discriminator_scheduler.step,
                gamma=self.training_cfg.discriminator_scheduler.gamma,
            ))

        return optimizers, schedulers

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch) -> None:
        # Unpack — see dataloader contract above
        x_true, x, observed_mask, artificial_mask = batch
        observed_mask = observed_mask.float()
        artificial_mask = artificial_mask.float()

        # The generator input mask is the observed entries MINUS the artificially
        # hidden ones. From the generator's perspective, artificial_mask positions
        # are "missing" — it must impute them, and we have clean labels for them.
        train_mask = observed_mask * (1 - artificial_mask)  # 1 = give to generator

        hint = generate_hint(train_mask.detach().cpu().numpy(), self.training_cfg.hint_rate)
        hint = torch.tensor(hint, dtype=torch.float32, device=x.device)

        batch = {
            "x": x,
            "x_true": x_true,
            "observed_mask": observed_mask,
            "train_mask": train_mask,
            "artificial_mask": artificial_mask,
            "hint": hint,
        }

        generator_optimizer, discriminator_optimizer = self.optimizers()

        # -- Generator forward pass --
        # Zero out artificially hidden positions so generator can't cheat
        x_masked = batch["train_mask"] * batch["x"]  # zero at artificial + originally missing positions

        x_hat = self.generator(x=x_masked, mask=batch["train_mask"])

        # Combine: keep real observed values (excluding hidden), fill rest with x_hat
        x_imputed = batch["train_mask"] * batch["x"] + (1 - batch["train_mask"]) * x_hat

        # -- Discriminator step --
        # Discriminator tries to distinguish truly observed from generated entries.
        # We detach x_imputed so discriminator gradients don't flow into generator.
        mask_hat = self.discriminator(x_imputed=x_imputed.detach(), hint=batch["hint"])
        discriminator_loss = discriminator_mask_loss(
            mask=batch["train_mask"], mask_hat=mask_hat, hint=batch["hint"]
        )
        discriminator_optimizer.zero_grad()
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()

        # -- Generator step --
        # Reconstruction loss: ONLY on artificially hidden positions where we
        # have clean ground-truth labels. This is the key fix — training on
        # observed positions gives a trivial signal (generator can just copy input).
        rmse = reconstruction_loss(
            x=batch["x_true"], x_hat=x_hat, mask=batch["artificial_mask"]
        )

        # Re-run discriminator (with gradients) for adversarial loss
        mask_hat = self.discriminator(x_imputed=x_imputed, hint=batch["hint"])
        adversarial_loss = generator_mask_loss(
            mask=batch["train_mask"], mask_hat=mask_hat, hint=batch["hint"]
        )

        total_loss = rmse + self.training_cfg.alpha * adversarial_loss
        generator_optimizer.zero_grad()
        self.manual_backward(total_loss)
        generator_optimizer.step()

        logger.debug(
            f"\n RMSE: {rmse}"
            f"\n Adversarial loss: {adversarial_loss}"
            f"\n Discriminator loss: {discriminator_loss}"
            f"\n Total loss: {total_loss}"
        )

        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/rmse", rmse)
        self.log("train/adversarial_loss", adversarial_loss)
        self.log("train/discriminator_loss", discriminator_loss)

    def on_train_epoch_end(self) -> None:
        for scheduler in self.lr_schedulers():
            scheduler.step()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch) -> None:
        x_true, x, observed_mask, artificial_mask = batch
        observed_mask = observed_mask.float()
        artificial_mask = artificial_mask.float()

        # During validation we additionally hide 10% of observed entries
        # on top of the existing artificial_mask, to get an imputation signal.
        # This mirrors the training setup: the generator sees train_mask,
        # and we evaluate reconstruction on the hidden subset.
        observed_bool = observed_mask.bool()
        extra_mask = torch.zeros_like(observed_mask)
        extra_mask[observed_bool] = (
            torch.rand(observed_bool.sum(), device=observed_mask.device) > 0.9
        ).float()

        # Combined evaluation mask: all artificially hidden positions
        eval_mask = (artificial_mask + extra_mask).clamp(max=1.0)

        # What the generator sees
        val_mask = observed_mask * (1 - eval_mask)

        hint = generate_hint(val_mask.detach().cpu().numpy(), self.training_cfg.hint_rate)
        hint = torch.tensor(hint, dtype=torch.float32, device=x.device)

        x_hat = self.generator(x=x, mask=val_mask)

        # Imputation RMSE: only on positions we deliberately hid (clean labels)
        imputation_rmse = reconstruction_loss(x=x_true, x_hat=x_hat, mask=eval_mask)

        x_imputed = val_mask * x + (1 - val_mask) * x_hat
        mask_hat = self.discriminator(x_imputed=x_imputed, hint=hint)

        discriminator_loss = discriminator_mask_loss(
            mask=val_mask, mask_hat=mask_hat, hint=hint
        )
        adversarial_loss = generator_mask_loss(
            mask=val_mask, mask_hat=mask_hat, hint=hint
        )
        rmse = reconstruction_loss(x=x_true, x_hat=x_hat, mask=eval_mask)
        total_loss = rmse + self.training_cfg.alpha * adversarial_loss

        logger.debug(
            f"\n Val imputation RMSE: {imputation_rmse}"
            f"\n Observed entries — predicted mean: {x_hat[observed_bool].mean().item():.4f}"
            f"\n Observed entries — true mean: {x_true[observed_bool].mean().item():.4f}"
        )

        self.log("val/total_loss", total_loss, prog_bar=True)
        self.log("val/rmse", rmse)
        self.log("val/imputation_rmse", imputation_rmse, prog_bar=True)
        self.log("val/adversarial_loss", adversarial_loss)
        self.log("val/discriminator_loss", discriminator_loss)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        logger.info("Started training...")
        early_stopping = EarlyStopping(
            monitor="val/imputation_rmse",
            min_delta=self.training_cfg.min_delta,
            patience=self.training_cfg.patience,
            verbose=False,
            mode="min",
            check_finite=True,
        )
        trainer = pl.Trainer(
            callbacks=early_stopping,
            max_epochs=self.training_cfg.num_epochs,
            accumulate_grad_batches=1,
        )
        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)

        if early_stopping.stopping_reason == EarlyStoppingReason.PATIENCE_EXHAUSTED:
            logger.info("Training stopped due to patience exhaustion")
        elif early_stopping.stopping_reason == EarlyStoppingReason.STOPPING_THRESHOLD:
            logger.info("Training stopped due to reaching stopping threshold")
        elif early_stopping.stopping_reason == EarlyStoppingReason.NOT_STOPPED:
            logger.info("Training completed normally without early stopping")

        if early_stopping.stopping_reason_message:
            logger.info(f"Details: {early_stopping.stopping_reason_message}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> torch.Tensor:
        """
        Returns raw generator output x_hat for every entry.
        The caller is responsible for combining with observed values via:
            x_imputed = observed_mask * x + (1 - observed_mask) * x_hat
        """
        self.eval()
        all_x_hat: list[torch.Tensor] = []

        for batch in loader:
            x, observed_mask = batch
            observed_mask = observed_mask.float()
            hint = generate_hint(observed_mask.detach().cpu().numpy(), self.training_cfg.hint_rate)
            hint = torch.tensor(hint, dtype=torch.float32, device=x.device)
            self.to(x.device)
            x_hat = self.generator(x=x, mask=observed_mask)
            all_x_hat.append(x_hat)

        return torch.cat(all_x_hat)

    @torch.no_grad()
    def impute(self, loader: DataLoader) -> torch.Tensor:
        """
        Returns the fully imputed matrix:
            observed entries  → original values kept
            missing entries   → generator predictions
        """
        self.eval()
        all_x_imputed: list[torch.Tensor] = []

        for batch in loader:
            x, observed_mask = batch
            observed_mask = observed_mask.float()
            hint = generate_hint(observed_mask.detach().cpu().numpy(), self.training_cfg.hint_rate)
            hint = torch.tensor(hint, dtype=torch.float32, device=x.device)
            self.to(x.device)
            x_hat = self.generator(x=x, mask=observed_mask)
            x_imputed = observed_mask * x + (1 - observed_mask) * x_hat
            all_x_imputed.append(x_imputed)

        return torch.cat(all_x_imputed)
