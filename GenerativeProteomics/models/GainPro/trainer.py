import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score

from models.GainPro.gain import Gain
from models.GainPro.metrics import Metrics
from utils.train_hypers import TrainHypers
from utils.data.helper import generate_hint
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
        self.discriminator_lr = train_hypers.generator_lr / 3
        self._init_optimizers()

        self.alpha = train_hypers.alpha
        self.hint_rate = train_hypers.hint_rate

        self.metrics = Metrics()

    def _init_optimizers(
        self,
    ) -> None:
        self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.generator_lr)
        self.optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.discriminator_lr)
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=35, gamma=0.5)
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=35, gamma=0.5)

    def _get_tissue_one_hot(
        self, 
        tissue_ids, 
        num_tissues
    ) -> torch.tensor:
        # todo adapt to groupkfold scenario
        # cause we will have a split a priori and then we will ended up with repeated classes
        return torch.nn.functional.one_hot(tissue_ids.long(), num_classes=num_tissues).float()

    def generate_sample(
        self,
        x, 
        mask,
        Z,
        tissue_one_hot,
    ):
        mask = mask.float()
        missing_data_with_noise = mask * x + (1 - mask) * Z
        input_G = torch.cat((missing_data_with_noise, mask), 1).float()
        # input_G = torch.cat((missing_data_with_noise, mask, tissue_one_hot), 1).float()
        return self.model.generator(input_G)

    def _loss_discriminator(
        self,
        x,
        mask,
        hint,
        Z,
        tissue_one_hot,
    ) -> torch.tensor:
        mask = mask.float()
        sample_G = self.generate_sample(x=x, mask=mask, Z=Z, tissue_one_hot=tissue_one_hot)
        x_hat = mask * x + (1 - mask) * sample_G
        # mask_hat = self.model.discriminator(torch.cat((x_hat.detach(), hint, tissue_one_hot), 1).float())
        mask_hat = self.model.discriminator(torch.cat((x_hat.detach(), hint), 1).float())

        # Only train on positions where hint == 0.5 (b_i = 0, i.e. unknown to discriminator)
        b_mask = (hint == 0.5)

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
        tissue_one_hot,
    ) -> torch.tensor:
        mask = mask.float()
        sample_G = self.generate_sample(x=x, mask=mask, Z=Z, tissue_one_hot=tissue_one_hot)
        x_hat = mask * x + (1 - mask) * sample_G

        mask_hat = self.model.discriminator(torch.cat((x_hat, hint), 1).float())
        # mask_hat = self.model.discriminator(torch.cat((x_hat, hint, tissue_one_hot), 1).float())

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
        return loss_G, loss_G_mse, loss_G_entropy

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
        print(f"  D grad norm: {d_norm:.4f}")
        self.optimizer_D.step()
        # for name, p in self.model.discriminator.named_parameters():
        #     print(f"  D {name} | mean: {p.data.mean():.6f} | std: {p.data.std():.6f}")
        #     break
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
        print(f"  G grad norm: {g_norm:.4f}")
        self.optimizer_G.step()
        # for name, p in self.model.generator.named_parameters():
        #     print(f"  G {name} | mean: {p.data.mean():.6f} | std: {p.data.std():.6f}")
        #     break
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

        x_hat = self.generate_sample(x=x, mask=mask, Z=Z, tissue_one_hot=tissue_one_hot)

        mse = nn.MSELoss()(x_true, x_hat)
        rmse = np.sqrt(mse.detach().cpu().numpy())

        if train_mode:
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
    ) -> None:
        tissue_one_hot = self._get_tissue_one_hot(tissue_ids, num_tissues).to(self.device)
        for ep in range(1, self.num_epochs+1):
            print(f"Epoch {ep}/{self.num_epochs}")

            hint = generate_hint(observed_mask.detach().cpu().numpy(), self.hint_rate)
            hint = torch.tensor(hint, device=self.device)

            discriminator_loss, generator_loss, generator_rmse, generator_entropy, rmse = self.epoch(
                x=x_train,
                x_true=x_train,
                mask=observed_mask,
                hint=hint,
                Z=torch.rand(x_train.shape, device=self.device),
                tissue_one_hot=tissue_one_hot,
                train_mode=True
            )

            self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss.item())
            self.metrics.train_metrics["generator_loss"].append(generator_loss.item())
            self.metrics.train_metrics["generator_rmse"].append(generator_rmse.item())
            self.metrics.train_metrics["generator_entropy"].append(generator_entropy.item())
            self.metrics.train_metrics["rmse"].append(rmse)

            self.scheduler_G.step()
            self.scheduler_D.step()

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
            x=x_missing, 
            mask=mask,
            Z=torch.zeros(x_missing.shape, device=self.device),
            tissue_one_hot=tissue_one_hot,
        )
        return x_hat.detach().cpu().numpy()

    def discriminate(
        self,
        x,
        mask,
        tissue_ids,
        num_tissues,
    ) -> torch.tensor:
        tissue_one_hot = self._get_tissue_one_hot(tissue_ids, num_tissues).to(self.device)
        hint = generate_hint(mask.detach().cpu().numpy(), hint_rate=0.0)
        hint = torch.tensor(hint, device=self.device)
        sample_G = self.generate_sample(
            x=x, 
            mask=mask, 
            Z=torch.rand(x.shape, device=self.device),
            tissue_one_hot=tissue_one_hot, 
        )
        mask = (~mask).float()
        fake_X = x * mask + sample_G * (1 - mask)
        # fake_input_D = torch.cat((fake_X.detach(), hint, tissue_one_hot), 1).float()
        fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
        logits = self.model.discriminator(fake_input_D)
        # print("Logits", logits)
        # print("Sigmoid", torch.sigmoid(logits))
        mask_hat = (torch.sigmoid(logits) > 0.5).float() # 1 = observed, 0 = missing
        # print("Predicted mask", mask_hat)
        # print("Artificial mask", mask)
        return mask_hat

    def compute_precision_recall_discriminator(
        self,
        x,
        mask: torch.tensor,
        observed_mask: torch.tensor,
        tissue_ids,
        num_tissues,
        positive_label: int,
    ) -> dict[str, float]:
        print(f"\n{'='*10}")
        print(f"Positive label: {positive_label} ({"Observed Entry" if positive_label==1 else "Missing Entry" if positive_label==0 else "Unknown"})")
        mask_hat = self.discriminate(
            x,
            mask,
            tissue_ids,
            num_tissues,
        )
        y_true = observed_mask.detach().cpu().numpy().ravel()
        y_pred = mask_hat.detach().cpu().numpy().ravel()
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
        print("Discriminator's precision:", precision)
        print("Discriminator's recall:", recall)
        print(f"{'='*10}")
        results = {"discriminator_precision": precision, "discriminator_recall": recall}
        return results