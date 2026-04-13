import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from models.GainPro.gain import Gain
from models.GainPro.metrics import Metrics
from utils.train_hypers import TrainHypers
from utils.data.dataset import Data, generate_hint
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

        self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.generator_lr)
        self.optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.discriminator_lr)

        self.alpha = train_hypers.alpha

        self.metrics = Metrics()

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
        loss = nn.BCEWithLogitsLoss(reduction="none")
        
        mask = mask.float()
        new_X = mask * x + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
   
        sample_G = self.model.generator(input_G)

        fake_X = new_X * mask + sample_G * (1 - mask)
        fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
        fake_Y = self.model.discriminator(fake_input_D)

        loss_D = (loss(fake_Y.float(), mask)).mean()

        # print("+++++++ loss discriminator", loss_D)

        return loss_D
    
    def _loss_generator(
        self,
        x,
        mask,
        hint,
        Z,
    ) -> torch.tensor:
        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        mask = mask.float()
        new_X = mask * x + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
        sample_G = self.model.generator(input_G)
        fake_X = new_X * mask + (1 - mask) * sample_G

        fake_input_D = torch.cat((fake_X, hint), 1).float()
        fake_Y = self.model.discriminator(fake_input_D)

        ones = torch.ones_like(x)
        # loss_G_entropy = (
        #     loss(fake_Y, ones.reshape(fake_Y.shape).float().to(self.device)) * (1 - mask)
        # ).mean()
        observed_entries = mask.bool()
        loss_G_entropy = (
            loss(fake_Y, ones.reshape(fake_Y.shape).float().to(self.device))[~observed_entries]
        ).mean()
        loss_G_mse = (
            loss_mse((sample_G[observed_entries]).float(), (x[observed_entries]).float())
        ).mean()

        loss_G = loss_G_entropy + self.alpha * loss_G_mse

        # print("------- loss generator", loss_G)
        # print("             loss g entropy", loss_G_entropy)
        # print("             loss g mse", loss_G_mse)

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
        idxs_folds: list = None,
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
            print("K-Fold strategy...")
            self.kfold_cv(
                num_folds=num_folds,
                idxs_folds=idxs_folds,
                data=data,
                experiment_writer=experiment_writer,
            )
        elif strategy == "group-k-fold":
            print("Stratified group k-fold strategy...")
            self.group_kfold_cv(
                num_folds=num_folds,
                data=data,
                experiment_writer=experiment_writer,
            )
        else:
            raise ValueError(
                f"Invalid cross validation strategy."
                f"Available strategies: Hold-out ('hold-out'), K-Fold ('k-fold') and Stratified Group K-Fold ('group-k-fold')."
            )
    
    def hold_out_cv(
        self,
        data: Data,
        experiment_writer: ExperimentWriter,
    ) -> None:
        experiment_writer.metadata_writer.set_out_dir(experiment_writer.evaluation_dir)
        experiment_writer.metadata_writer.set_start_time(datetime.now())
        # train
        for ep in range(1, self.num_epochs+1):
            print(f"Epoch {ep}/{self.num_epochs}")

            x = data.missing.detach().clone()
            x_true = data.reference.detach().clone()
            observed_mask = data.observed_mask.detach().clone()
            hint = data.hint.detach().clone()
            Z = torch.rand(x.shape, device=self.device)

            discriminator_loss, generator_loss, rmse = self.epoch(
                x=x,
                x_true=x_true,
                mask=observed_mask,
                hint=hint,
                Z=Z,
                train_mode=True
            )

            self.metrics.train_metrics["discriminator_loss"].append(discriminator_loss.item())
            self.metrics.train_metrics["generator_loss"].append(generator_loss.item())
            self.metrics.train_metrics["rmse"].append(rmse)

        experiment_writer.metadata_writer.set_end_time(datetime.now())
        evaluation_dir = experiment_writer.evaluation_dir / "hold-out"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(out_dir=evaluation_dir)
        experiment_writer.metadata_writer.save_metadata()

        # Evaluate (only on artificial masked entries=0)
        artificial_missing_mask = data.artificial_missing_mask.detach().clone() 
        x_hat = self.generate_sample(
            data=x, 
            mask=artificial_missing_mask
        )
        mse_loss = nn.MSELoss(reduction="sum")
        artificial_missing_mask_np = (~artificial_missing_mask).float()
        mse = (
            (mse_loss(x_true * artificial_missing_mask_np, x_hat * artificial_missing_mask_np) / artificial_missing_mask_np.sum())  # ~artificial missing mask to compare only the masked entries
        ).mean()
        rmse = np.sqrt(mse.detach().cpu().numpy())

        # Invert the normalization
        max_norm = data.max_norm.values
        min_norm = data.min_norm.values
        x_hat = x_hat.detach().cpu().numpy() * (max_norm - min_norm) + min_norm
        x_true = x_true.detach().cpu().numpy() * (max_norm - min_norm) + min_norm

        # Invert the log2(x + 1) from dataset_builder.py: x = 2^y - 1
        x_hat_log2p1_inverse = np.power(2, x_hat)-1

        # Since we filled NANs entries with zeros
        observed_mask_np = observed_mask.detach().cpu().numpy()
        x_true_log2p1_inverse = np.where(observed_mask_np == 0, np.nan, np.power(2, x_true) - 1)

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
        print("RMSE:", rmse.item())

        self.metrics.val_metrics["discriminator_loss"].append(None)
        self.metrics.val_metrics["generator_loss"].append(None)
        self.metrics.val_metrics["rmse"].append(rmse.item())

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
            # Train
            for ep in range(1, self.num_epochs+1):
                print(f"Epoch {ep}/{self.num_epochs} \n")

                x_train = missing_train.detach().clone()
                x_true_train = reference_train.detach().clone()
                observed_mask_train = observed_mask_train.detach().clone()
                hint_train = hint_train.detach().clone()
                Z_train = torch.rand(x_train.shape, device=self.device)

                train_discriminator_loss, train_generator_loss, train_rmse = self.epoch(
                    x=x_train,
                    x_true=x_true_train,
                    mask=observed_mask_train,
                    hint=hint_train,
                    Z=Z_train,
                    train_mode=True,
                )

                self.metrics.train_metrics["discriminator_loss"].append(train_discriminator_loss.item())
                self.metrics.train_metrics["generator_loss"].append(train_generator_loss.item())
                self.metrics.train_metrics["rmse"].append(train_rmse.item())

                x_val = missing_val.detach().clone()
                x_true_val = reference_val.detach().clone()
                observed_mask_val = observed_mask_val.detach().clone()
                hint_val = hint_val.detach().clone()
                Z_val = torch.rand(x_val.shape, device=self.device)

                val_discriminator_loss, val_generator_loss, val_rmse = self.epoch(
                    x=x_val,
                    x_true=x_true_val,
                    mask=observed_mask_val,
                    hint=hint_val,
                    Z=Z_val,
                    train_mode=False,
                )

                self.metrics.val_metrics["discriminator_loss"].append(val_discriminator_loss.item())
                self.metrics.val_metrics["generator_loss"].append(val_generator_loss.item())
                self.metrics.val_metrics["rmse"].append(val_rmse.item())

            experiment_writer.metadata_writer.set_end_time(datetime.now())
            experiment_writer.metadata_writer.set_fold_id(fold_id=fold_id)
            experiment_writer.metadata_writer.save_metadata()
            experiment_writer.metrics_writer.log_metrics(metrics=self.metrics, fold_id=fold_id)
            
            # Evaluate
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

            print("type test idx", type(test_idx), test_idx)

            experiment_writer.result_writer.save_predictions(
                fold_id=fold_id,
                sample_ids=np.array(data.sample_names)[test_idx],
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
        """
        Stratified Group K-Fold cross-validation for evaluating whether GAIN can
        reconstruct a sample it has never seen before.
        """
        sgkf = StratifiedGroupKFold(n_splits=num_folds)
        groups = data.tissue.cpu().numpy()

        y = groups
        for fold_id, (trainval_idx, test_idx) in enumerate(
            sgkf.split(X=data.reference.cpu().numpy(), y=y, groups=groups), start=1
        ):
            print(f"\n\n------------ Fold {fold_id}/{num_folds} ------------\n")

            model = Gain(
                input_dim=data.reference.shape[1],
                num_hidden_layers_generator=self.model.num_hidden_layers_generator,
                num_hidden_layers_discriminator=self.model.num_hidden_layers_discriminator,
            )
            self.model = model
            self.model.to(self.device)

            print("Learning rate generator", self.generator_lr)
            print("Learning rate discriminator", self.discriminator_lr)
            self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.generator_lr)
            self.optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.discriminator_lr)

            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, random_state=42)

            x_train_ref = data.reference[train_idx, :]
            observed_mask_train = data.observed_mask[train_idx, :]
            # hint_train = data.hint[train_idx, :]

            x_val_ref = data.reference[val_idx, :]
            observed_mask_val = data.observed_mask[val_idx, :]
            # hint_val = data.hint[val_idx, :]

            kfold_dir = experiment_writer.evaluation_dir / "groupkfold"
            kfold_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.metadata_writer.set_out_dir(kfold_dir)
            experiment_writer.metadata_writer.set_start_time(datetime.now())

            num_train_samples, num_proteins = x_train_ref.shape[0], x_train_ref.shape[1]
            num_val_samples = x_val_ref.shape[0]

            for ep in range(1, self.num_epochs + 1):
                print(f"Epoch {ep}/{self.num_epochs}")

                Z_train = torch.rand((num_train_samples, num_proteins), device=self.device)
                Z_val = torch.rand((num_val_samples, num_proteins), device=self.device)

                hint = generate_hint(data.observed_mask.detach().cpu().numpy(), data.hint_rate)
                hint = torch.from_numpy(hint).to(self.device)

                hint_train = hint[train_idx, :]
                hint_val = hint[val_idx, :]

                train_discriminator_loss, train_generator_loss, train_rmse = self.epoch(
                    x=x_train_ref.detach().clone(),
                    x_true=x_train_ref.detach().clone(),  # ground truth = same ref (no artificial noise)
                    mask=observed_mask_train.detach().clone(),
                    hint=hint_train.detach().clone(),
                    Z=Z_train.detach().clone(),
                    train_mode=True,
                )

                self.metrics.train_metrics["discriminator_loss"].append(train_discriminator_loss.item())
                self.metrics.train_metrics["generator_loss"].append(train_generator_loss.item())
                self.metrics.train_metrics["rmse"].append(train_rmse.item())

                val_discriminator_loss, val_generator_loss, val_rmse = self.epoch(
                    x=x_val_ref.detach().clone(),
                    x_true=x_val_ref.detach().clone(),  # ground truth = same ref (no artificial noise)
                    mask=observed_mask_val.detach().clone(),
                    hint=hint_val.detach().clone(),
                    Z=Z_val.detach().clone(),
                    train_mode=False,
                )

                self.metrics.val_metrics["discriminator_loss"].append(val_discriminator_loss.item())
                self.metrics.val_metrics["generator_loss"].append(val_generator_loss.item())
                self.metrics.val_metrics["rmse"].append(val_rmse.item())

            experiment_writer.metadata_writer.set_end_time(datetime.now())
            experiment_writer.metadata_writer.set_fold_id(fold_id=fold_id)
            experiment_writer.metadata_writer.save_metadata()
            experiment_writer.metrics_writer.log_metrics(metrics=self.metrics, fold_id=fold_id)
            self.metrics.init_fold()

            x_true_test = data.reference[test_idx, :]
            observed_mask_test = data.observed_mask[test_idx, :]

            with torch.no_grad():
                x_hat = self.generate_sample(
                    data=x_true_test,         # real observed values
                    mask=observed_mask_test   # real observed mask
                )
                print(f"X hat {torch.mean(x_hat, dim=0)}, {torch.std(x_hat, dim=0)} (mean ± std)")
                print(f"X true {torch.mean(x_true_test, dim=0)}, {torch.std(x_true_test, dim=0)} (mean ± std)")
                mse = nn.MSELoss(reduction="none")(  #todo why none and not mean here directly?
                    x_true_test[observed_mask_test], 
                    x_hat[observed_mask_test]
                ).mean()
                rmse = np.sqrt(mse.detach().cpu().numpy())

            # --- invert normalization ---
            max_norm = data.max_norm.values
            min_norm = data.min_norm.values
            x_hat_np = x_hat.detach().cpu().numpy() * (max_norm - min_norm) + min_norm
            x_true_np = x_true_test.detach().cpu().numpy() * (max_norm - min_norm) + min_norm

            # --- invert log2(x+1) transform---
            x_hat_original = np.power(2, x_hat_np) - 1

            observed_mask_test_np = observed_mask_test.detach().cpu().numpy()
            x_true_original = np.where(
                observed_mask_test_np == 0,
                np.nan,
                np.power(2, x_true_np) - 1,
            )

            print("normalized and transformed x_hat mean:", np.mean(x_hat_original, axis=0))
            print("normalized and transformed x_true mean:", np.nanmean(x_true_original, axis=0))

            experiment_writer.result_writer.save_predictions(
                fold_id=fold_id,
                sample_ids=np.array(data.sample_names)[test_idx],
                feature_names=data.feature_names,
                true_values=x_true_original,
                pred_values=x_hat_original,
                observed_mask=observed_mask_test,
                artificial_missing_mask=observed_mask_test,
                group_ids=data.tissue[test_idx].cpu().numpy(),
                group_mapping=data.tissue_mapping,
            )

            experiment_writer.result_writer.save_test_rmse(
                out_dir=kfold_dir,
                rmse=rmse,
                fold_id=fold_id,
            )
            print(f"Test RMSE (fold {fold_id}): {rmse:.4f}")
            print(f"Test MSE (fold {fold_id}): {mse:.4f}")

            experiment_writer.split_writer.save_fold_splits(
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=test_idx,
            )