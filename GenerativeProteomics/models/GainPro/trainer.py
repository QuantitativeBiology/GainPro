import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, train_test_split

from utils.data.dataset import Data
from models.GainPro.gain import Gain
from models.GainPro.metrics import Metrics
from utils.train_hypers import TrainHypers
from utils.data.proteomics_scaler import ProteomicsScaler
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
        self.batch_size = train_hypers.batch_size

        self.generator_lr = train_hypers.generator_lr
        self.discriminator_lr = train_hypers.discriminator_lr

        self.alpha = train_hypers.alpha

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
    
    def _loss_discriminator(
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
        hint,
        epoch, #todo test if interval training results in better training
    ) -> tuple[torch.tensor, torch.tensor, float]:
        num_samples, num_proteins = x.shape[0], x.shape[1]
        Z = torch.rand((num_samples, num_proteins), device=self.device) * 0.01 #todo 0.01 can be a hyperparameter (represents noise, confirm tho)

        # interval training
        # if epoch % 20 == 0:
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

    def fit(
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
        strategy: str,
        data: Data,
        idxs_folds: list=None,
        num_folds: int = None,
        experiment_writer: ExperimentWriter = None,
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
                idxs_folds=idxs_folds,
                # idxs_folds=[{"train_val": [0,1], "test": [2,3]}], #todo fix
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
        
        # print("reference", data.reference)
        # print("missing", data.missing)
        # print("observed mask", data.observed_mask)
        # print("artificial missing mask", data.artificial_missing_mask)

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
                hint=hint,
                epoch=ep,
            )

            self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss.item())
            self.metrics.train_metrics["generator_loss"].append(generator_loss.item())
            self.metrics.train_metrics["rmse"].append(rmse)

        experiment_writer.metadata_writer.set_end_time(datetime.now())
        evaluation_dir = experiment_writer.evaluation_dir / "hold-out"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(out_dir=evaluation_dir)
        experiment_writer.metadata_writer.save_metadata()

        # evaluate (only on artificially masked entries=0)
        artificial_missing_mask = data.artificial_missing_mask.detach().clone() # only on artificially masked entries
        x_hat = self.generate_sample(
            data=x, 
            mask=artificial_missing_mask
        )
        mse_loss = nn.MSELoss(reduction="none")
        mse = (
            mse_loss(x_true * ~artificial_missing_mask, x_hat * ~artificial_missing_mask)
        ).mean()
        rmse = np.sqrt(mse.detach().cpu().numpy())

        # Invert the normalization
        max_norm = data.max_norm.values
        min_norm = data.min_norm.values
        x_hat = x_hat.detach().cpu().numpy() * (max_norm - min_norm) + min_norm # inverse normalization
        x_true = x_true.detach().cpu().numpy() * (max_norm - min_norm) + min_norm # inverse normalization

        # debugging purposes
        print("Real mean:", x_true.mean(0))
        print("Fake mean:", x_hat.mean(0))
        print("Real std:", x_true.std(0))
        print("Fake std:", x_hat.std(0))

        # Invert the log2(x + 1) from dataset_builder.py: x = 2^y - 1
        x_hat_log2p1_inverse = np.power(2, x_hat)-1

        # Since we filled NANs entries with zeros
        observed_mask_np = observed_mask.detach().cpu().numpy()
        x_true_log2p1_inverse = np.where(observed_mask_np == 0, np.nan, np.power(2, x_true) - 1)

        # print("\n\n\n\n =================== ")
        # print("true values", x_true_log2p1_inverse)
        # print("predicted values", x_hat_log2p1_inverse)
        # print("observed mask", observed_mask)
        # print("artificial missing mask", artificial_missing_mask)
        # print("data missing", data.missing)
        # print("observed mask & artificial missing mask",  torch.logical_and(observed_mask, artificial_missing_mask))
        # print("\n\n\n\n =================== ")

        experiment_writer.result_writer.save_predictions(
            sample_ids=data.sample_names,
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

    
    def kfold_cv(
        self,
        num_folds: int,
        idxs_folds: list,
        data: Data,
        experiment_writer: ExperimentWriter,
    ) -> None:

        for fold_id in range(1, num_folds+1):
            print(f"\n\n------------ Fold {fold_id}/{num_folds} ------------\n")

            trainval_idx = idxs_folds[fold_id-1]["trainval_idx"]
            test_idx = idxs_folds[fold_id-1]["test_idx"]

            # Create new model
            model = Gain(
                input_dim=data.missing.shape[1],
                num_hidden_layers_generator=self.model.num_hidden_layers_generator,
                num_hidden_layers_discriminator=self.model.num_hidden_layers_discriminator,
            )
            self.model = model
            self.model.to(self.device)

            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.2)

            missing_train, missing_val = data.missing[train_idx, :], data.missing[val_idx, :]
            reference_train, reference_val = data.reference[train_idx, :], data.reference[val_idx, :]
            observed_mask_train, observed_mask_val = data.observed_mask[train_idx, :], data.observed_mask[val_idx, :]
            hint_train, hint_val = data.hint[train_idx, :], data.hint[val_idx, :]

            kfold_dir = experiment_writer.evaluation_dir / "kfold"
            kfold_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.metadata_writer.set_out_dir(kfold_dir)
            experiment_writer.metadata_writer.set_start_time(datetime.now())
            # train
            for ep in range(1, self.num_epochs+1):
                print(f"Epoch {ep}/{self.num_epochs} \n")

                x_train = missing_train.detach().clone()
                x_true_train = reference_train.detach().clone()
                observed_mask_train = observed_mask_train.detach().clone()
                hint_train = hint_train.detach().clone()

                train_discriminator_loss, train_generator_loss, train_rmse = self.epoch(
                    x=x_train,
                    x_true=x_true_train,
                    mask=observed_mask_train,
                    hint=hint_train,
                    epoch=ep,
                )

                self.metrics.train_metrics["discriminator_loss"].append(train_discriminator_loss.item())
                self.metrics.train_metrics["generator_loss"].append(train_generator_loss.item())
                self.metrics.train_metrics["rmse"].append(train_rmse.item())

                x_val = missing_val.detach().clone()
                x_true_val = reference_val.detach().clone()
                observed_mask_val = observed_mask_val.detach().clone()
                hint_val = hint_val.detach().clone()

                val_discriminator_loss, val_generator_loss, val_rmse = self.epoch(
                    x=x_val,
                    x_true=x_true_val,
                    mask=observed_mask_val,
                    hint=hint_val,
                    epoch=ep,
                )

                self.metrics.val_metrics["discriminator_loss"].append(val_discriminator_loss.item())
                self.metrics.val_metrics["generator_loss"].append(val_generator_loss.item())
                self.metrics.val_metrics["rmse"].append(val_rmse.item())

            experiment_writer.metadata_writer.set_end_time(datetime.now())
            experiment_writer.metadata_writer.set_fold_id(fold_id=fold_id)
            experiment_writer.metadata_writer.save_metadata()
            experiment_writer.metrics_writer.log_metrics(metrics=self.metrics, fold_id=fold_id)
            
            # evaluate
            x_test = data.missing[test_idx, :]
            x_true_test = data.reference[test_idx, :]
            observed_mask_test = data.observed_mask[test_idx, :]
            artificial_missing_mask_test = data.artificial_missing_mask[test_idx, :]

            x_hat = self.generate_sample(
                data=x_test, 
                mask=artificial_missing_mask_test
            )
            print("x hat", x_hat)
            mse_loss = nn.MSELoss(reduction="none")
            mse = (
                mse_loss(x_true_test * ~artificial_missing_mask_test, x_hat * ~artificial_missing_mask_test)
            ).mean()
            rmse = np.sqrt(mse.detach().cpu().numpy())

            # Invert the normalization #todo these following steps should ideally be a different function
            max_norm = data.max_norm.values
            min_norm = data.min_norm.values
            x_hat = x_hat.detach().cpu().numpy() * (max_norm - min_norm) + min_norm # inverse normalization
            x_true_test = x_true_test.detach().cpu().numpy() * (max_norm - min_norm) + min_norm # inverse normalization

            # Invert the log2(x + 1) from dataset_builder.py: x = 2^y - 1
            x_hat_log2p1_inverse = np.power(2, x_hat)-1

            # Since we filled NANs entries with zeros
            observed_mask_np = observed_mask_test.detach().cpu().numpy()
            x_true_log2p1_inverse = np.where(observed_mask_np == 0, np.nan, np.power(2, x_true_test) - 1)

            experiment_writer.result_writer.save_predictions(
                fold_id=fold_id,
                sample_ids=data.sample_names[test_idx], #todo verificar
                feature_names=data.feature_names,
                true_values=x_true_log2p1_inverse,
                pred_values=x_hat_log2p1_inverse,
                observed_mask=observed_mask_test,
                artificial_missing_mask=artificial_missing_mask_test,
            )

            evaluation_dir = experiment_writer.evaluation_dir / "kfold"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.result_writer.save_test_rmse(
                out_dir=evaluation_dir,
                rmse=rmse,
                fold_id=fold_id
            )
            print("test rmse", rmse)

            experiment_writer.split_writer.save_fold_splits(
                fold_id=fold_id,
                train_idx=trainval_idx,
                test_idx=test_idx,
            )

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
            model = Gain(
                input_dim=data.missing.shape[1],
                num_hidden_layers_generator=self.model.num_hidden_layers_generator,
                num_hidden_layers_discriminator=self.model.num_hidden_layers_discriminator,
            )
            self.model = model
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