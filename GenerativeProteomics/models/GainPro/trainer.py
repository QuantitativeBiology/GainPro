import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold

from utils.data.dataset import Data
from models.GainPro.gain import Gain
from models.GainPro.metrics import Metrics
from utils.data.proteomics_scaler import ProteomicsScaler
from utils.writers.experiment_writer import ExperimentWriter

class Trainer:
    def __init__(
        self,
        model: Gain,
        num_epochs: int,
        generator_lr: float,
        discriminator_lr: float,
        alpha: float
    ) -> "Trainer":
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model

        self.num_epochs = num_epochs

        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.alpha = alpha

        self.metrics = Metrics()

    def generate_sample(
        self,
        data, 
        mask
    ):
        size, dim = data.shape[0], data.shape[1]

        Z = torch.rand((size, dim), device=self.device) * 0.01

        mask = mask.int()
        missing_data_with_noise = mask * data + (1 - mask) * Z
        input_G = torch.cat((missing_data_with_noise, mask), 1).float()

        return self.model.generator(input_G)

    def _update_discriminator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.tensor:
        loss = nn.BCEWithLogitsLoss(reduction="none")

        optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.discriminator_lr)
        
        mask = mask.float()
        new_X = mask * x + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
   
        sample_G = self.model.generator(input_G)

        fake_X = new_X * mask + sample_G * (1 - mask)
        fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
        fake_Y = self.model.discriminator(fake_input_D)

        loss_D = (loss(fake_Y.float(), mask)).mean()
        loss_D_aux = loss_D.detach().clone()

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        return loss_D_aux
        
    def _update_generator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.tensor:
        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.generator_lr)

        ones = torch.ones_like(x)

        mask = mask.float()
        new_X = mask * x + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
        sample_G = self.model.generator(input_G)
        fake_X = new_X * mask + sample_G * (1 - mask)

        fake_input_D = torch.cat((fake_X, hint), 1).float()
        fake_Y = self.model.discriminator(fake_input_D)

        loss_G_entropy = (
            loss(fake_Y, ones.reshape(fake_Y.shape).float().to(self.device)) * (1 - mask)
        ).mean()
        loss_G_mse = (
            loss_mse((sample_G * mask).float(), (x * mask).float())
        ).mean()

        loss_G = loss_G_entropy + self.alpha * loss_G_mse
        loss_G_aux = loss_G.detach().clone()

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        return loss_G_aux
    
    def epoch(
        self,
        x,
        x_true,
        mask,
        hint
    ) -> tuple[torch.tensor, torch.tensor, float]:
        num_samples, num_proteins = x.shape[0], x.shape[1]
        Z = torch.rand((num_samples, num_proteins), device=self.device) * 0.01 #todo 0.01 can be a hyperparameter (represents noise, confirm tho)
        # print("z shape", Z.shape)
        # print("x shape", x.shape)
        # print("mask shape", mask.shape)


        discriminator_loss = self._update_discriminator(
            x=x,
            mask=mask,
            hint=hint,  
            Z=Z
        )
        generator_loss = self._update_generator(
            x=x,
            mask=mask,
            hint=hint,
            Z=Z
        )

        x_hat = self.generate_sample(
            data=x, 
            mask=mask
        )

        mse_loss = nn.MSELoss(reduction="none")
        mse = (
            mse_loss(x_true * mask, x_hat * mask)
        ).mean()

        rmse = np.sqrt(mse.detach().cpu().numpy())

        return discriminator_loss, generator_loss, rmse

    def train(
        self,
        data: Data,
    ) -> None:
        self.model.train()
    
        for ep in range(1, self.num_epochs+1):
            print(f"Epoch {ep}/{self.num_epochs} \n")
            
            x = data.reference.detach().clone()
            x_true = data.missing.detach().clone()
            mask = data.observed_mask.detach().clone()
            hint = data.hint.detach().clone()

            discriminator_loss, generator_loss, rmse = self.epoch(
                x=x,
                x_true=x_true,
                mask=mask,
                hint=hint
            )

            self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss.item())
            self.metrics.train_metrics["generator_loss"].append(generator_loss.item())
            self.metrics.train_metrics["rmse"].append(rmse.item())
        
    def evaluate(
        self,
        data: Data,
        strategy: str,
        experiment_writer: ExperimentWriter = None,
        num_folds: int = None,
    ) -> None: #todo move the evaluate into an evaluation.py file
        """
        Args:
            - strategy (str): Cross validation strategies. Available: Hold-out ("hold-out"), K-Fold ("k-fold") and Stratified Group K-Fold ('group-k-fold').
        """
        if strategy == "hold-out":
            print("Hold-out strategy...")
            self.hold_out_cv(
                data=data,
                experiment_writer=experiment_writer,
            )
        elif strategy == "k-fold":
            self.kfold_cv(
                num_folds=num_folds,
                data=data,
                experiment_writer=experiment_writer,
            )
        elif strategy == "group-k-fold":
            self.group_kfold_cv(
                num_folds=num_folds,
                data=data,
                experiment_writer=experiment_writer,
            )
        else:
            raise ValueError(f"Invalid cross validation strategy. Available strategies: Hold-out ('hold-out'), K-Fold ('k-fold') and Stratified Group K-Fold ('group-k-fold').")
    
    def hold_out_cv(
        self,
        data: Data,
        experiment_writer: ExperimentWriter,
    ) -> None:
        
        print("reference", data.reference)
        print("missing", data.missing)
        print("observed mask", data.observed_mask)

        # scaler = ProteomicsScaler()
        
        experiment_writer.metadata_writer.set_out_dir(experiment_writer.evaluation_dir)
        experiment_writer.metadata_writer.set_start_time(datetime.now())
        # train
        for ep in range(1, self.num_epochs+1):
            print(f"Epoch {ep}/{self.num_epochs} \n")

            x = data.missing.detach().clone()
            x_true = data.reference.detach().clone()
            observed_mask = data.observed_mask.detach().clone()
            hint = data.hint.detach().clone()

            # scaler.fit(x, mask)
            # x = torch.from_numpy(x.copy()).to(self.device)
            # x_true = torch.from_numpy(x_true.copy()).to(self.device)

            discriminator_loss, generator_loss, rmse = self.epoch(
                x=x,
                x_true=x_true,
                mask=observed_mask,
                hint=hint
            )

            self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss.item())
            self.metrics.train_metrics["generator_loss"].append(generator_loss.item())
            self.metrics.train_metrics["rmse"].append(rmse)

        experiment_writer.metadata_writer.set_end_time(datetime.now())
        evaluation_dir = experiment_writer.evaluation_dir / "hold-out"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(out_dir=evaluation_dir)
        experiment_writer.metadata_writer.save_metadata()

        # evaluate (only on artificially masked entries)
        artificial_missing_mask = data.artificial_missing_mask.detach().clone() # only on artificially masked entries
        print("mask mean", observed_mask.detach().numpy().mean())
        x_hat = self.generate_sample(
            data=x, 
            mask=artificial_missing_mask
        )
        print("x hat normalized", x_hat)
        mse_loss = nn.MSELoss(reduction="none")
        mse = (
            mse_loss(x_true * artificial_missing_mask, x_hat * artificial_missing_mask)
        ).mean()
        rmse = np.sqrt(mse.detach().cpu().numpy())

        x_hat_aux = x_hat.detach().cpu().numpy()
        max_norm = data.max_norm.values
        min_norm = data.min_norm.values
        print("\n X HAT AUX", x_hat_aux)
        print("data max norm", type(max_norm), type(min_norm), type(x_hat_aux))
        x_hat = x_hat_aux * (max_norm - min_norm) + min_norm # inverse normalization
        print("x hat 'renormalized'", x_hat)

        # debugging purposes
        # print("Real mean:", x_true.mean(0))
        # print("Fake mean:", x_hat.mean(0))
        # print("Real std:", x_true.std(0))
        # print("Fake std:", x_hat.std(0))

        # revert the logarithm (log2(x+1))
        x_true_log2p1_inverse = np.power(2, x_true.detach().cpu().numpy())-1
        x_hat_log2p1_inverse = np.power(2, x_hat)-1

        # print("\n\n\n")
        # print("X TRUE", x_true_log2p1_inverse)
        # print("X HAT", x_hat_log2p1_inverse)
        # print("\n\n\n")

        experiment_writer.result_writer.save_predictions(
            sample_ids=np.arange(start=0, stop=data.reference.shape[0]),
            feature_names=data.feature_names,
            true_values=x_true_log2p1_inverse,
            pred_values=x_hat_log2p1_inverse,
            observed_mask=observed_mask,
            artificial_missing_mask=artificial_missing_mask,
        )

        experiment_writer.result_writer.save_test_rmse(
            out_dir=evaluation_dir,
            rmse=rmse.item(),
        )

        self.metrics.test_metrics["discriminator_loss"].append(None)
        self.metrics.test_metrics["generator_loss"].append(None)
        self.metrics.test_metrics["rmse"].append(rmse.item())

        experiment_writer.metrics_writer.log_metrics(
            metrics=self.metrics,
        )

        print("test rmse", self.metrics.test_metrics["rmse"][0])

    
    def kfold_cv(
        self,
        num_folds: int,
        data: Data,
        experiment_writer: ExperimentWriter,
    ) -> None:
        
        scaler = ProteomicsScaler()
        kf = KFold(n_splits=num_folds, shuffle=True)

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(data.missing), start=1):
            print(f"\n\n------------ Fold {fold_id}/{num_folds} ------------\n")

            # Create new model
            self.model = Gain(input_dim=data.missing.shape[1])
            self.model.to(self.device)

            missing_train = data.missing[train_idx, :]
            reference_train = data.reference[train_idx, :]
            mask_train = data.observed_mask[train_idx, :]
            hint_train = data.hint[train_idx, :]

            # scaler.fit(reference_train, mask_train)
            # missing_train = torch.from_numpy(scaler.transform(missing_train.detach().cpu().numpy()).copy()).to(self.device)
            # reference_train = torch.from_numpy(scaler.transform(reference_train.detach().cpu().numpy()).copy()).to(self.device)

            # print("missing log and scaled", missing_train)
            # print("reference log and scaled", reference_train)

            kfold_dir = experiment_writer.evaluation_dir / "kfold"
            kfold_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.metadata_writer.set_out_dir(kfold_dir)
            experiment_writer.metadata_writer.set_start_time(datetime.now())
            # train
            for ep in range(1, self.num_epochs+1):
                print(f"Epoch {ep}/{self.num_epochs} \n")

                x = missing_train.detach().clone()
                x_true = reference_train.detach().clone()
                mask = mask_train.detach().clone()
                hint = hint_train.detach().clone()

                discriminator_loss, generator_loss, rmse = self.epoch(
                    x=x,
                    x_true=x_true,
                    mask=mask,
                    hint=hint
                )

                self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss)
                self.metrics.train_metrics["generator_loss"].append(generator_loss)
                self.metrics.train_metrics["rmse"].append(rmse)
            experiment_writer.metadata_writer.set_end_time(datetime.now())
            experiment_writer.metadata_writer.save_metadata()
            
            # evaluate
            x_test = data.missing[test_idx, :]
            x_true_test = data.reference[test_idx, :]
            artificial_missing_mask = data.artificial_missing_mask[test_idx, :]
            
            # x_test = torch.from_numpy(scaler.transform(x_test.detach().cpu().numpy()).copy()).to(self.device)
            # x_true_test = torch.from_numpy(scaler.transform(x_true_test.detach().cpu().numpy()).copy()).to(self.device)
            #todo quando for no impute, predict a sério, tenho de dar revert do log com np.power(2, x_value) - 1

            x_hat = self.generate_sample(
                data=x_test, 
                mask=artificial_missing_mask
            )
            # x_hat = torch.from_numpy(scaler.inverse_transform(x_hat.detach().cpu().numpy()).copy()).to(self.device)
            # x_true_test = torch.from_numpy(scaler.transform(x_true_test.detach().cpu().numpy()).copy()).to(self.device)
            print("x hat", x_hat)
            mse_loss = nn.MSELoss(reduction="none")
            mse = (
                mse_loss(x_true_test * artificial_missing_mask, x_hat * artificial_missing_mask)
            ).mean()
            rmse = np.sqrt(mse.detach().cpu().numpy())

            experiment_writer.result_writer.save_predictions(
                out_dir=experiment_writer.preds_dir,
                fold_id=fold_id,
                sample_ids=test_idx,
                feature_names=data.feature_names,
                true_values=x_true_test.detach().cpu().numpy(),
                pred_values=x_hat.detach().cpu().numpy(),
                observed_mask=mask,
                artificial_missing_mask=artificial_missing_mask,
            )

            evaluation_dir = experiment_writer.evaluation_dir / "kfold"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.result_writer.save_test_rmse(
                out_dir=evaluation_dir,
                rmse=rmse,
                fold_id=fold_id
            )
            print("test rmse", rmse)

    def group_kfold_cv(
        self,
        num_folds: int,
        data: Data,
        experiment_writer: ExperimentWriter,
    ) -> None:
        #todo repetir com outras seeds porque podemos ter tido sorte no split que aconteceu

        scaler = ProteomicsScaler()
        
        # n_cell_lines = len(torch.unique(data.cell_line))
        # n_splits = min(num_folds, n_cell_lines) -> not realistic since the cell lines can be in the hundreds
        gkf = GroupKFold(n_splits=num_folds, shuffle=True)
        groups = data.cell_line

        for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X=data.missing.cpu().numpy(), groups=groups.cpu().numpy()), start=1):
            print(f"\n\n------------ Fold {fold_id}/{num_folds} ------------\n")

            # Create new model
            self.model = Gain(input_dim=data.missing.shape[1])
            self.model.to(self.device)

            missing_train = data.missing[train_idx, :]
            reference_train = data.reference[train_idx, :]
            mask_train = data.observed_mask[train_idx, :]
            hint_train = data.hint[train_idx, :]

            scaler.fit(reference_train, mask_train)
            missing_train = torch.from_numpy(scaler.transform(missing_train.detach().cpu().numpy()).copy()).to(self.device)
            reference_train = torch.from_numpy(scaler.transform(reference_train.detach().cpu().numpy()).copy()).to(self.device)

            kfold_dir = experiment_writer.evaluation_dir / "group-kfold"
            kfold_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.metadata_writer.set_out_dir(kfold_dir)
            experiment_writer.metadata_writer.set_start_time(datetime.now())
            # train
            for ep in range(1, self.num_epochs+1):
                print(f"Epoch {ep}/{self.num_epochs} \n")

                x = missing_train.detach().clone()
                x_true = reference_train.detach().clone()
                mask = mask_train.detach().clone()
                hint = hint_train.detach().clone()

                discriminator_loss, generator_loss, rmse = self.epoch(
                    x=x,
                    x_true=x_true,
                    mask=mask,
                    hint=hint
                )

                self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss)
                self.metrics.train_metrics["generator_loss"].append(generator_loss)
                self.metrics.train_metrics["rmse"].append(rmse)
            experiment_writer.metadata_writer.set_end_time(datetime.now())
            experiment_writer.metadata_writer.save_metadata()
            
            # evaluate
            x_test = data.missing[test_idx, :]
            x_true_test = data.reference[test_idx, :]
            artificial_missing_mask = data.artificial_missing_mask[test_idx, :]

            x_test = torch.from_numpy(scaler.transform(x_test.detach().cpu().numpy()).copy()).to(self.device)
            x_true_test = torch.from_numpy(scaler.transform(x_true_test.detach().cpu().numpy()).copy()).to(self.device)

            x_hat = self.generate_sample(
                data=x_test, 
                mask=artificial_missing_mask
            )
            print("x hat", x_hat)
            mse_loss = nn.MSELoss(reduction="none")
            mse = (
                mse_loss(x_true_test * artificial_missing_mask, x_hat * artificial_missing_mask)
            ).mean()
            rmse = np.sqrt(mse.detach().cpu().numpy())

            group_ids_test = data.cell_line[test_idx]

            preds_dir = experiment_writer.preds_dir / "group-kfold"
            preds_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.result_writer.save_predictions(
                out_dir=preds_dir,
                fold_id=fold_id,
                sample_ids=test_idx,
                feature_names=data.feature_names,
                true_values=x_true_test.detach().cpu().numpy(),
                pred_values=x_hat.detach().cpu().numpy(),
                observed_mask=mask,
                artificial_missing_mask=artificial_missing_mask,
                group_mapping=data.cell_line_mapping,
                group_ids=group_ids_test,
            )
            evaluation_dir = experiment_writer.evaluation_dir / "group-kfold"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.result_writer.save_test_rmse(
                out_dir=evaluation_dir,
                rmse=rmse,
                fold_id=fold_id
            )
            print("test rmse", rmse)