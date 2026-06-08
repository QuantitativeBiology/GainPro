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
        mask_hat = self.discriminator(x_imputed=batch["x"], hint=batch["hint"])
        return {"x_hat": x_hat, "mask_hat": mask_hat}
    
    def _compute_losses(
        self,
        batch,
        out,
    ) -> dict:
        # binary cross entropy generator and discriminator
        rmse = reconstruction_loss(x=batch["x"], x_hat=out["x_hat"], mask=batch["mask"])

        discriminator_entropy = 0 #todo não sei muito bem quando é que deve atualizar.

        discriminator_loss = discriminator_mask_loss(
            mask=batch["mask"], 
            mask_hat=out["mask_hat"], 
            hint=batch["hint"]
        )

        return {
            "rmse": rmse,
            "discriminator_entropy": discriminator_entropy,
            "discriminator_loss": discriminator_loss
        }
    
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
        batch = {"x": x, "x_true": x_true, "mask": mask, "hint": hint}

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
        # Generator adversarial loss: fool discriminator on missing positions
        # gen_adversarial = -torch.mean(
        #     (1 - batch["mask"]) * torch.log(mask_hat + 1e-8)
        # )
        gen_adversarial = generator_mask_loss(mask=batch["mask"], mask_hat=mask_hat, hint=batch["hint"])
        total_loss = rmse + self.training_cfg.alpha * gen_adversarial
        logger.debug(
            f"\n RMSE: {rmse}"
            f"\n Adversarial loss: {gen_adversarial}"
            f"\n Total loss: {total_loss}"
        )
        generator_optimizer.zero_grad()
        self.manual_backward(total_loss)
        generator_optimizer.step()

        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/rmse", rmse)
        self.log("train/discriminator_entropy", gen_adversarial)
        self.log("train/discriminator_loss", discriminator_loss)
    
    def training_step_original(
        self, 
        batch,
    ) -> torch.Tensor:
        x_true, x, mask = batch
        mask = mask.float()
        hint = generate_hint(mask.detach().cpu().numpy(), self.training_cfg.hint_rate)
        hint = torch.tensor(hint, dtype=torch.float32).to(mask.device)
        batch = {"x": x, "x_true": x_true, "mask": mask, "hint": hint}
        logger.debug(
            f"\n Batch: {batch}"
            f"\n Hint device: {hint.device}"
        )

        optimizers = self.optimizers()

        generator_optimizer, discriminator_optimizer = optimizers[0], optimizers[1]

        out = self.forward(batch)
        losses = self._compute_losses(batch=batch, out=out)

        discriminator_optimizer.zero_grad()
        self.manual_backward(losses["discriminator_loss"])
        discriminator_optimizer.step()

        total_loss = losses["rmse"] - self.training_cfg.alpha * losses["discriminator_entropy"]

        generator_optimizer.zero_grad()
        self.manual_backward(total_loss)
        generator_optimizer.step()

        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/rmse", losses["rmse"])
        self.log("train/discriminator_entropy", losses["discriminator_entropy"])
        self.log("train/discriminator_loss", losses["discriminator_loss"])

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
        batch = {"x": x, "x_true": x_true, "mask": mask, "hint": hint}

        x_hat = self.generator(x=batch["x"], mask=batch["mask"])
        x_imputed = mask * x + (1 - mask) * x_hat

        mask_hat = self.discriminator(x_imputed=x_imputed, hint=batch["hint"])
        discriminator_loss = discriminator_mask_loss(mask=batch["mask"], mask_hat=mask_hat, hint=batch["hint"])

        # --- Generator loss ---
        rmse = reconstruction_loss(x=batch["x"], x_hat=x_hat, mask=batch["mask"])
        gen_adversarial = generator_mask_loss(mask=batch["mask"], mask_hat=mask_hat, hint=batch["hint"])
        total_loss = rmse + self.training_cfg.alpha * gen_adversarial

        self.log("val/total_loss", total_loss, prog_bar=True)
        self.log("val/rmse", rmse)
        self.log("val/gen_adversarial", gen_adversarial)
        self.log("val/discriminator_loss", discriminator_loss)

    def validation_step_original(
        self,
        batch,
    ) -> None:
        x_true, x, mask = batch
        mask = mask.float()
        hint = generate_hint(mask.detach().cpu().numpy(), self.training_cfg.hint_rate)
        hint = torch.tensor(hint, dtype=torch.float32).to(mask.device)
        batch = {"x": x, "x_true": x_true, "mask": mask, "hint": hint}
        logger.debug(
            f"\n Batch: {batch}"
            f"\n Hint device: {hint.device}"
        )
        out = self.forward(batch)
        losses = self._compute_losses(batch=batch, out=out)

        total_loss = losses["rmse"] - self.training_cfg.alpha * losses["discriminator_entropy"]
        self.log("val/total_loss", total_loss, prog_bar=True)
        self.log("val/rmse", losses["rmse"])
        self.log("val/discriminator_entropy", losses["discriminator_entropy"])
        self.log("val/discriminator_loss", losses["discriminator_loss"])

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
            batch = {"x": x, "mask": mask, "hint": hint}
            out = self.forward(batch)
            x_pred = out["x_hat"].cpu()
            all_x_pred.append(x_pred.cpu())

        return torch.cat(all_x_pred)

    @torch.no_grad()
    def impute(
        self,
        loader: DataLoader,
    ) -> torch.Tensor:
        pass