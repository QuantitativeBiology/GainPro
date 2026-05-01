import torch
import numpy as np
import torch.nn as nn

from models.GainPro.gain import Gain
from models.GainPro.metrics import Metrics
from utils.train_hypers import TrainHypers
from utils.data.dataset import generate_hint

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
        self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.generator_lr)
        self.optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.discriminator_lr)

    def generate_sample(
        self,
        data, 
        mask
    ):
        size, dim = data.shape[0], data.shape[1]

        Z = torch.rand((size, dim), device=self.device)

        mask = mask.int()
        missing_data_with_noise = mask * data + (1 - mask) * Z
        input_G = torch.cat((missing_data_with_noise, mask), 1).float()

        return self.model.generator(input_G)
    
    def _loss_discriminator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.tensor:
        loss = nn.BCELoss(reduction="none")
        
        mask = mask.float()
        new_X = mask * x + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
   
        sample_G = self.model.generator(input_G)

        fake_X = new_X * mask + sample_G * (1 - mask)
        fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
        fake_Y = self.model.discriminator(fake_input_D)

        loss_D = (loss(fake_Y.float(), mask)).mean()
        return loss_D
    
    def _loss_generator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.tensor:
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        mask = mask.float()
        new_X = mask * x + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
        sample_G = self.model.generator(input_G)
        fake_X = new_X * mask + (1 - mask) * sample_G

        fake_input_D = torch.cat((fake_X, hint), 1).float()
        fake_Y = self.model.discriminator(fake_input_D)

        ones = torch.ones_like(x)
        observed_entries = mask.bool()
        loss_G_entropy = (
            loss(fake_Y, ones.reshape(fake_Y.shape).float().to(self.device))[~observed_entries]
        ).mean()
        loss_G_mse = (
            loss_mse((sample_G[observed_entries]).float(), (x[observed_entries]).float())
        ).mean()

        loss_G = loss_G_entropy + self.alpha * loss_G_mse
        return loss_G
    
    def _update_discriminator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.tensor:
        loss_D = self._loss_discriminator(
            x,
            mask,
            hint,
            Z
        )
        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()
        return loss_D
    
    def _update_generator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.tensor:
        loss_G = self._loss_generator(
            x,
            mask,
            hint,
            Z
        )
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G
    
    def epoch(
        self,
        x,
        x_true,
        Z,
        mask,
        hint,
        train_mode: bool = True,
    ) -> tuple[torch.tensor, torch.tensor, float]:
        if train_mode:
            self.model.train()
            discriminator_loss = self._update_discriminator(x=x, mask=mask, hint=hint, Z=Z)
            generator_loss = self._update_generator(x=x, mask=mask, hint=hint, Z=Z)
        else:
            self.model.eval()
            with torch.no_grad():
                discriminator_loss = self._loss_discriminator(x=x, mask=mask, hint=hint, Z=Z)
                generator_loss = self._loss_generator(x=x, mask=mask, hint=hint, Z=Z)

        x_hat = self.generate_sample(data=x, mask=mask)

        observed = mask.bool()
        mse = nn.MSELoss()(x_true[observed], x_hat[observed])
        rmse = np.sqrt(mse.detach().cpu().numpy())

        if not train_mode:
            self.model.train()

        return discriminator_loss.detach().clone(), generator_loss.detach().clone(), rmse

    def train(
        self,
        x_train: torch.tensor,
        observed_mask: torch.tensor,
        hint_rate: float=0.5,
    ) -> None:
        for ep in range(1, self.num_epochs+1):
            print(f"Epoch {ep}/{self.num_epochs}")

            Z = torch.rand(x_train.shape, device=self.device)

            hint = generate_hint(observed_mask, hint_rate).to(self.device)

            discriminator_loss, generator_loss, rmse = self.epoch(
                x=x_train,
                x_true=x_train,
                mask=observed_mask,
                hint=hint,
                Z=Z,
                train_mode=True
            )

            self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss.item())
            self.metrics.train_metrics["generator_loss"].append(generator_loss.item())
            self.metrics.train_metrics["rmse"].append(rmse)
    
    def impute(
        self,
        x_missing,
        mask,
    ) -> np.ndarray:
        x_hat = self.generate_sample(
            data=x_missing, 
            mask=mask,
        )
        return x_hat.detach().cpu().numpy()