import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

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
    def __init__(
        self,
        input_dim: int,
        hypers: GainConfig,
        training_cfg: GainTrainingConfig,
        generator_output_activation: nn.Module=None,
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
            num_hidden_layers=hypers.num_hidden_layers_discriminator
        )

        self.training_cfg = training_cfg

    def forward(
        self,
        batch,
    ) -> dict:
        x_hat = self.generator(x=batch["x"], mask=batch["mask"])
        mask = batch["mask"]
        x_imputed = mask * batch["x"] + (1 - mask) * x_hat
        mask_hat = self.discriminator(x_imputed=x_imputed, hint=batch["hint"])
        return {"x_hat": x_hat, "mask_hat": mask_hat}
    
    def configure_optimizers(
        self
    ):
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
            generator_scheduler = torch.optim.lr_scheduler.StepLR(
                generator_optimizer,
                step_size=self.training_cfg.generator_scheduler.step,
                gamma=self.training_cfg.generator_scheduler.gamma,
            )
            schedulers.append(generator_scheduler)
        if self.training_cfg.discriminator_scheduler is not None:
            discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
                discriminator_optimizer,
                step_size=self.training_cfg.discriminator_scheduler.step,
                gamma=self.training_cfg.discriminator_scheduler.gamma,
            )
            schedulers.append(discriminator_scheduler)

        return optimizers, schedulers
    
    def training_step(
        self, 
        batch,
    ) -> torch.Tensor:
        x_true, x, mask = batch
        mask = mask.float()
        hint = generate_hint(mask.detach().cpu().numpy(), self.training_cfg.hint_rate)
        hint = torch.tensor(hint, dtype=torch.float32).to(mask.device)
        # Add noise
        noise = torch.randn_like(x) * 0.1
        x_noisy = mask * x + (1 - mask) * (x + noise)
        batch = {"x": x_noisy, "x_true": x_true, "mask": mask, "hint": hint}

        optimizers = self.optimizers()

        generator_optimizer, discriminator_optimizer = optimizers[0], optimizers[1]

        x_hat = self.generator(x=batch["x"], mask=batch["mask"])

        # Discriminator
        x_imputed = mask * batch["x"] + (1 - mask) * x_hat
        mask_hat = self.discriminator(x_imputed=x_imputed.detach(), hint=batch["hint"])
        discriminator_loss = discriminator_mask_loss(mask=batch["mask"], mask_hat=mask_hat, hint=batch["hint"])
        logger.debug(f"\n Discriminator loss: {discriminator_loss}")
        discriminator_optimizer.zero_grad()
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()

        # Generator
        rmse = reconstruction_loss(x=batch["x"], x_hat=x_hat, mask=batch["mask"])
        mask_hat = self.discriminator(x_imputed=x_imputed, hint=batch["hint"])
        adversarial_loss = generator_mask_loss(mask=batch["mask"], mask_hat=mask_hat, hint=batch["hint"])
        total_loss = rmse + self.training_cfg.alpha * adversarial_loss
        logger.debug(
            f"\n RMSE: {rmse}"
            f"\n Adversarial loss: {adversarial_loss}"
            f"\n Total loss: {total_loss}"
        )
        generator_optimizer.zero_grad()
        self.manual_backward(total_loss)
        generator_optimizer.step()

        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/rmse", rmse)
        self.log("train/adversarial_loss", adversarial_loss)
        self.log("train/discriminator_loss", discriminator_loss)

    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        for scheduler in schedulers:
            scheduler.step()

    def validation_step(
        self, 
        batch
    ) -> None:
        x_true, x, mask = batch
        mask = mask.float()
        hint = generate_hint(mask.detach().cpu().numpy(), self.training_cfg.hint_rate)
        hint = torch.tensor(hint, dtype=torch.float32).to(mask.device)
        noise = torch.randn_like(x) * 0.1
        x_noisy = mask * x + (1 - mask) * (x + noise)
        batch = {"x": x_noisy, "x_true": x_true, "mask": mask, "hint": hint}

        x_hat = self.generator(x=batch["x"], mask=batch["mask"])
        x_imputed = mask * x + (1 - mask) * x_hat

        mask_hat = self.discriminator(x_imputed=x_imputed, hint=batch["hint"])
        discriminator_loss = discriminator_mask_loss(mask=batch["mask"], mask_hat=mask_hat, hint=batch["hint"])

        rmse = reconstruction_loss(x=batch["x"], x_hat=x_hat, mask=batch["mask"])
        adversarial_loss = generator_mask_loss(mask=batch["mask"], mask_hat=mask_hat, hint=batch["hint"])
        total_loss = rmse + self.training_cfg.alpha * adversarial_loss

        # Debug purposes
        observed_mask = batch["mask"].bool()
        missing_mask = ~observed_mask
        logger.debug(
            f"\n Observed entries mean"
            f"\n    X predicted: {x_hat[observed_mask].mean().item()}"
            f"\n    X true: {batch["x"][observed_mask].mean().item()}"
            f"\n Missing entries mean"
            f"\n    X predicted: {x_hat[missing_mask].mean().item()}"
            f"\n    X true: {batch["x"][missing_mask].mean().item()}"

        )

        self.log("val/total_loss", total_loss, prog_bar=True)
        self.log("val/rmse", rmse)
        self.log("val/adversarial_loss", adversarial_loss)
        self.log("val/discriminator_loss", discriminator_loss)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None=None,
    ) -> None:
        logger.info("Started training...")
        trainer = pl.Trainer(max_epochs=self.training_cfg.num_epochs, accumulate_grad_batches=1)
        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
    ) -> torch.Tensor:
        self.eval()

        all_x_pred: list[torch.Tensor] = []
        
        for batch in loader:
            x, mask = batch
            mask = mask.float()
            hint = generate_hint(mask.detach().cpu().numpy(), self.training_cfg.hint_rate)
            hint = torch.tensor(hint, dtype=torch.float32).to(mask.device)
            self.to(mask.device)
            batch = {"x": x, "mask": mask, "hint": hint}
            out = self.forward(batch)
            x_pred = out["x_hat"]
            all_x_pred.append(x_pred)

        return torch.cat(all_x_pred)

    @torch.no_grad()
    def impute(
        self,
        loader: DataLoader,
    ) -> torch.Tensor:
        pass