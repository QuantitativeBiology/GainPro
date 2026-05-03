import torch
import numpy as np
import torch.nn as nn

from models.GainPro.gain import Gain
from models.GainPro.metrics import Metrics
from utils.train_hypers import TrainHypers
from utils.data.dataset import generate_hint
from utils.writers.experiment_writer import ExperimentWriter

class Trainer:
    def __init__(
        self,
        model: Gain,
        train_hypers: TrainHypers,
    ) -> "Trainer":
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model

        self.num_epochs = train_hypers.num_epochs

        self.generator_lr = train_hypers.generator_lr
        self.discriminator_lr = train_hypers.discriminator_lr
        self._init_optimizers()

        self.alpha = train_hypers.alpha

        self.metrics = Metrics()

    def _init_optimizers(
        self,
    ) -> None:
        self.optimizer_G = torch.optim.Adam(
            self.model.generator.parameters(), 
            lr=self.generator_lr
        )
        self.optimizer_D = torch.optim.Adam(
            self.model.discriminator.parameters(), 
            lr=self.discriminator_lr
        )

    def _get_tissue_one_hot(
        self, 
        tissue_ids, 
        num_tissues
    ) -> torch.tensor:
        return torch.nn.functional.one_hot(tissue_ids.long(), num_classes=num_tissues).float()

    def generate_sample(
        self,
        data, 
        mask,
        tissue_one_hot,
        Z=None,
    ):
        size, dim = data.shape[0], data.shape[1]

        if Z is None:
            Z = torch.rand((size, dim), device=self.device)

        mask = mask.float()
        missing_data_with_noise = mask * data + (1 - mask) * Z
        input_G = torch.cat((missing_data_with_noise, mask, tissue_one_hot), 1).float()
        return self.model.generator(input_G)
    
    def _loss_discriminator(
        self,
        x,
        mask,
        hint,
        Z,
        tissue_one_hot,
    ) -> torch.tensor:
        loss = nn.BCEWithLogitsLoss(reduction="none")
        
        mask = mask.float()
        new_X = mask * x + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
   
        sample_G = self.generate_sample(data=x, mask=mask, tissue_one_hot=tissue_one_hot, Z=Z)

        fake_X = new_X * mask + sample_G * (1 - mask)
        fake_input_D = torch.cat((fake_X.detach(), hint, tissue_one_hot), 1).float()
        fake_Y = self.model.discriminator(fake_input_D)

        loss_D = (loss(fake_Y, mask)).mean()
        return loss_D
    
    def _loss_generator(
        self,
        x,
        mask,
        hint,
        Z,
        tissue_one_hot,
    ) -> torch.tensor:
        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        mask = mask.float()
        new_X = mask * x + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
        # sample_G = self.model.generator(input_G)
        sample_G = self.generate_sample(data=x, mask=mask, tissue_one_hot=tissue_one_hot, Z=Z)
        fake_X = new_X * mask + (1 - mask) * sample_G

        fake_input_D = torch.cat((fake_X, hint, tissue_one_hot), 1).float()
        fake_Y = self.model.discriminator(fake_input_D)

        ones = torch.full_like(fake_Y, 1.0, device=self.device)
        observed_entries = mask.bool()
        loss_G_entropy = (
            loss(fake_Y, ones.reshape(fake_Y.shape).float().to(self.device))[~observed_entries]
        ).mean()
        loss_G_mse = (
            loss_mse((sample_G[observed_entries]).float(), (x[observed_entries]).float())
        ).mean()

        loss_G = loss_G_entropy + self.alpha * loss_G_mse
        return loss_G, loss_G_mse, loss_G_mse
    
    def _update_discriminator(
        self,
        x,
        mask,
        hint,
        Z,
        tissue_one_hot,
    ) -> torch.tensor:
        loss_D = self._loss_discriminator(
            x=x,
            mask=mask,
            hint=hint,
            Z=Z,
            tissue_one_hot=tissue_one_hot,
        )
        self.optimizer_D.zero_grad()
        loss_D.backward()
        d_norm = sum(p.grad.norm().item() ** 2
            for p in self.model.discriminator.parameters()
            if p.grad is not None) ** 0.5
        # print(f"  D grad norm: {d_norm:.4f}")
        # torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=0.1)
        self.optimizer_D.step()
        return loss_D
    
    def _update_generator(
        self,
        x,
        mask,
        hint,
        Z,
        tissue_one_hot,
    ) -> torch.tensor:
        loss_G, generator_rmse, generator_entropy = self._loss_generator(
            x=x,
            mask=mask,
            hint=hint,
            Z=Z,
            tissue_one_hot=tissue_one_hot,
        )
        self.optimizer_G.zero_grad()
        loss_G.backward()
        g_norm = sum(p.grad.norm().item() ** 2
            for p in self.model.generator.parameters()
            if p.grad is not None) ** 0.5
        # print(f"  G grad norm: {g_norm:.4f}")
        # torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_norm=0.1)
        self.optimizer_G.step()
        return loss_G, generator_rmse, generator_entropy
    
    def epoch(
        self,
        x,
        x_true,
        Z,
        mask,
        hint,
        tissue_one_hot,
        train_mode: bool=True,
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, float]:
        if train_mode:
            self.model.train()
            discriminator_loss = self._update_discriminator(x=x, mask=mask, hint=hint, Z=Z, tissue_one_hot=tissue_one_hot)
            generator_loss, generator_rmse, generator_entropy = self._update_generator(x=x, mask=mask, hint=hint, Z=Z, tissue_one_hot=tissue_one_hot)
        else:
            self.model.eval()
            with torch.no_grad():
                discriminator_loss = self._loss_discriminator(x=x, mask=mask, hint=hint, Z=Z, tissue_one_hot=tissue_one_hot)
                generator_loss, generator_rmse, generator_entropy = self._loss_generator(x=x, mask=mask, hint=hint, Z=Z, tissue_one_hot=tissue_one_hot)

        x_hat = self.generate_sample(data=x, mask=mask, tissue_one_hot=tissue_one_hot, Z=Z)

        observed = mask.bool()
        mse = nn.MSELoss()(x_true[~observed], x_hat[~observed])
        rmse = np.sqrt(mse.detach().cpu().numpy())

        if not train_mode:
            self.model.train()

        return (
            discriminator_loss.detach().clone(), 
            generator_loss.detach().clone(), 
            generator_rmse.detach().clone(), 
            generator_entropy.detach().clone(), 
            rmse
        )
    
    def train(
        self,
        x_train: torch.tensor,
        observed_mask: torch.tensor,
        tissue_ids,
        num_tissues,
        experiment_writer: ExperimentWriter,
        hint_rate: float=0.5,
    ) -> None:
        tissue_one_hot = self._get_tissue_one_hot(tissue_ids, num_tissues).to(self.device)
        for ep in range(1, self.num_epochs+1):
            print(f"Epoch {ep}/{self.num_epochs}")

            Z = torch.rand(x_train.shape, device=self.device)

            hint = generate_hint(observed_mask.detach().cpu().numpy(), hint_rate)
            hint = torch.tensor(hint, device=self.device)

            discriminator_loss, generator_loss, generator_rmse, generator_entropy, rmse = self.epoch(
                x=x_train,
                x_true=x_train,
                mask=observed_mask,
                hint=hint,
                Z=Z,
                tissue_one_hot=tissue_one_hot,
                train_mode=True
            )

            self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss.item())
            self.metrics.train_metrics["generator_loss"].append(generator_loss.item())
            self.metrics.train_metrics["generator_rmse"].append(generator_rmse.item())
            self.metrics.train_metrics["generator_entropy"].append(generator_rmse.item())
            self.metrics.train_metrics["rmse"].append(rmse)

        experiment_writer.metrics_writer.log_metrics(metrics=self.metrics)
        
    def impute(
        self,
        x_missing,
        mask,
        tissue_ids,
        num_tissues,
    ) -> np.ndarray:
        tissue_one_hot = self._get_tissue_one_hot(tissue_ids, num_tissues).to(self.device)
        x_hat = self.generate_sample(
            data=x_missing, 
            mask=mask,
            tissue_one_hot=tissue_one_hot,
        )
        return x_hat.detach().cpu().numpy()