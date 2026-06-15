import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
    EarlyStoppingReason
)
from torch.utils.data import DataLoader

from models.AutoEncoder.encoder import Encoder
from models.AutoEncoder.decoder import Decoder
from models.AutoEncoder.losses import reconstruction_loss
from utils.configs.model_config import AutoEncoderConfig
from utils.configs.training_config import AutoEncoderTrainingConfig

logger = logging.getLogger(__name__)

class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hypers: AutoEncoderConfig,
        training_cfg: AutoEncoderTrainingConfig,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hypers.hidden_dims,
            latent_dim=hypers.latent_dim,
        )

        self.decoder = Decoder(
            output_dim=input_dim,
            hidden_dims=hypers.hidden_dims,
            latent_dim=hypers.latent_dim,
        )

        self.training_cfg = training_cfg
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.encoder(x)
        return self.decoder(embedding)
    
    def compute_loss(
        self,
        x,
        x_hat,
        mask,
    ) -> torch.Tensor:
        rmse = reconstruction_loss(x=x, x_hat=x_hat, mask=mask)
        return rmse

    def training_step(
        self,
        batch,
    ) -> torch.Tensor:
        x_true, x, mask = batch
        out = self(x)
        logger.debug(
            f"\n Batch: {x}"
            f"\n X true: {x_true}"
            f"\n Out: {out}"
        )
        loss = self.compute_loss(x=x_true, x_hat=out, mask=mask)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int):
        x_true, x, mask = batch
        out = self.forward(x)
        loss = self.compute_loss(x=x_true, x_hat=out, mask=mask)
        logger.debug(
            f"\n Validation loss: {loss}"
        )
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
    
    def configure_optimizers(self):
        """Set up Adam optimizers and StepLR schedulers."""
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )
        opt = torch.optim.Adam(params=params, lr=self.training_cfg.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=self.training_cfg.scheduler.step,
            gamma=self.training_cfg.scheduler.gamma,
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        logger.info("Started training...")
        early_stopping = EarlyStopping(
            monitor="val/loss", 
            min_delta=self.training_cfg.min_delta, 
            patience=self.training_cfg.patience, 
            verbose=False,
            mode="min",
            check_finite=True,
        )
        trainer = pl.Trainer(
            callbacks=early_stopping,
            max_epochs=self.training_cfg.num_epochs, 
            accumulate_grad_batches=1
        )
        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Check why training stopped
        if early_stopping.stopping_reason == EarlyStoppingReason.PATIENCE_EXHAUSTED:
            logger.info("Training stopped due to patience exhaustion")
        elif early_stopping.stopping_reason == EarlyStoppingReason.STOPPING_THRESHOLD:
            logger.info("Training stopped due to reaching stopping threshold")
        elif early_stopping.stopping_reason == EarlyStoppingReason.NOT_STOPPED:
            logger.info("Training completed normally without early stopping")

        if early_stopping.stopping_reason_message:
            logger.info(f"Details: {early_stopping.stopping_reason_message}")

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
    ) -> torch.Tensor:
        """
        Predict all entries.

        Args:
        - loader (DataLoader): PyTorch DataLoader yielding:
            - batch (torch.Tensor): Input tensor containing observed data (and possibly masked values).

        Returns:
        - (torch.Tensor): Predicted values (model predictions).
        """
        self.eval()

        all_x_pred: list[torch.Tensor] = []
        
        for batch in loader:
            x_pred = self.forward(batch)
            all_x_pred.append(x_pred.cpu())

        return torch.cat(all_x_pred)
    
    @torch.no_grad()
    def impute(
        self,
        loader: DataLoader,
    ) -> torch.Tensor:
        """
        Perform missing-value imputation.

        Args:
        - loader (DataLoader): PyTorch DataLoader yielding:
            - batch (torch.Tensor): Input tensor containing observed data (and possibly masked values).

        Returns:
        - (torch.Tensor): Imputed values (model predictions).
        """
        self.eval()

        all_x_hat: list[torch.Tensor] = []
        
        for batch, mask in loader:
            mask = mask.float()
            x_hat = self.forward(batch)
            x_hat = (1 - mask) * batch + mask * x_hat
            all_x_hat.append(x_hat.cpu())

        return torch.cat(all_x_hat)