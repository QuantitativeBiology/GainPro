import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

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
        batch,
        x_hat,
        mask,
    ) -> torch.Tensor:
        rmse = reconstruction_loss(x=batch, x_hat=x_hat, mask=mask)
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
        loss = self.compute_loss(batch=x_true, x_hat=out, mask=mask)
        return loss
    
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
        trainer = pl.Trainer(max_epochs=self.training_cfg.num_epochs)
        trainer.fit(self, train_dataloaders=train_loader)
    
    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
    ) -> torch.Tensor:
        self.eval()

        all_x_out: list[torch.Tensor] = []
        
        for batch, mask in loader:
            mask = mask.float()
            x_hat = self.forward(batch)
            x_out = (1 - mask) * batch + mask * x_hat
            all_x_out.append(x_out.cpu())

        return torch.cat(all_x_out).numpy()