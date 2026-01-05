import time
from datetime import datetime
import warnings
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# model
from gainpro.gain_dann_model import GainDann
from gainpro.hypers import Params
from gainpro.dataset import generate_hint
from gainpro.output import Metrics

from gainpro.data_utils import Data
from gainpro.params_gain_dann import ParamsGainDann
from gainpro.early_stopping import EarlyStopping
from gainpro.metrics import MetricsTracker
from gainpro.evaluation import EvaluationTracker
from gainpro.dann_utils import save_model

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# so why split the train into a different file from the model?
# this is tricky.
# 1. the model is different from the training strategy
# taking into consideration this argument it will be way easier
# to now access different strategies (as a matter of fact that was
# were i got lost almost every single time)

class GainDannTrain:
    def __init__(self, data: Data, hypers: ParamsGainDann, early_stop_patience: int=30, save_dir: str=None, save_model: bool=False):
        self.data = data

        self.hypers = hypers
        self.early_stop_patience = early_stop_patience

        if save_dir is None:
            timestamp = datetime.now().strftime('%d-%m_%H:%M')
            self.save_dir = f"./checkpoints/{timestamp}"
        else:
            self.save_dir = save_dir

        self.save = save_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def initialize_model(self) -> GainDann:
        input_dim = self.data.n_proteins
        gain_params = Params()
        gain_metrics = Metrics(gain_params)
        protein_names = self.data.protein_names

        model = GainDann(protein_names, input_dim, latent_dim=input_dim, n_class=self.data.n_projects, num_hidden_layers=self.hypers["num_hidden_layers"], dann_params=self.hypers, gain_params=gain_params, gain_metrics=gain_metrics)
        model.encoder.apply(self.init_weights)
        model.decoder.apply(self.init_weights)
        model.to(self.device)

        logger.info("\n GAIN-DANN model created.\n")
        logger.debug(model)

        return model

    def initialize_optimizer_scheduler(self, model):
        optimizer = torch.optim.AdamW([
                    {'params': model.encoder.parameters()},
                    {'params': model.domain_classifier.parameters()},
                    {'params': model.decoder.parameters()}
                    ],
                    lr=self.hypers["learning_rate"],
                    weight_decay=self.hypers["weight_decay"]
                )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
        )
        return optimizer, scheduler

    def set_model_mode(self, model, is_training=True):
        if is_training:
            model.encoder.train()
            model.decoder.train()
            model.domain_classifier.train()
        else:
            model.encoder.eval()
            model.decoder.eval()
            model.domain_classifier.eval()

    def preprocess_batch(self, x_missing, x_domain, x_true):
        
        x_missing, x_domain, x_true = x_missing.to(self.device), x_domain.to(self.device), x_true.to(self.device)

        x_filled = x_missing.clone()
        x_filled[torch.isnan(x_filled)] = 0

        # masks
        mask_missing = (~torch.isnan(x_missing)).float().to(self.device)
        hint = generate_hint(mask_missing, self.hypers["hint_rate"]).to(self.device)
        mask_true = (~torch.isnan(x_true)).float().to(self.device)

        x_ground_truth = x_true.clone().detach()
        x_ground_truth[torch.isnan(x_ground_truth)] = 0

        return x_filled, x_domain, x_ground_truth, mask_missing, mask_true, hint

    def epoch(self, model, dataloader, optimizer, p: float, metrics: MetricsTracker, is_training: bool=True):
        """ 
            - p (int): Training progress. #todo used in formula paper, improve
        """
        # torch.autograd.set_detect_anomaly(True) test purposes

        # Losses
        domain_criterion = nn.CrossEntropyLoss()
        loss_gain = nn.BCELoss(reduction="none")
        loss_mse_gain = nn.MSELoss(reduction="mean")

        domain_accuracy_epoch = 0
        gain_loss_epoch = 0      
        domain_loss_epoch = 0
        total_loss_epoch = 0 # model loss
        reconstruction_loss_epoch = 0

        for data in dataloader:

            with torch.set_grad_enabled(is_training):
                
                x_missing, x_domain, x_true = data
                x_filled, x_domain, x_ground_truth, mask_missing, mask_true, hint = self.preprocess_batch(x_missing, x_domain, x_true)

                # Encoder
                z = model.encoder(x_filled) # z is the latent representation (= x encoded)
                lambda_max = 1.0
                lambda_dann = lambda_max * (2. / (1 + np.exp(-10 * p)) - 1) # the paper sets y equal to 10 (empirical)
                # lambda_dann = 1.0
                z_grl = model.grl(z, lambd=lambda_dann)

                # Domain classifier
                x_domain_hat = model.domain_classifier(z_grl) # prediction of x domain
                domain_loss = domain_criterion(x_domain_hat, x_domain)
                domain_loss_epoch += domain_loss.detach().cpu().numpy().mean()
                domain_pred_labels = torch.argmax(x_domain_hat, dim=1)
                domain_accuracy = (domain_pred_labels == x_domain).float().mean().item()
                domain_accuracy_epoch += domain_accuracy

                # GAIN training (as function `model.py/train`)
                n_samples, dim = z.shape[0], z.shape[1]
                Z = torch.rand((n_samples, dim), device=self.device) * 0.01
                
                z_gain = z.clone()  # keeps device + dtype
                # if you want encoder gradients to flow:
                z_gain.requires_grad_(True)

                if is_training:
                    model.gain._update_D(z_gain.detach(), mask_missing, hint, Z, loss_gain)
                    model.gain._update_G(z_gain.detach(), mask_missing, hint, Z, loss_gain)

                samples = model.gain.generate_sample(z_gain, mask_missing)

                loss_mse = loss_mse_gain(mask_missing * z_gain, mask_missing * samples)
                gain_loss_epoch += loss_mse.detach().cpu().numpy()
            
                x_imputed = z * mask_missing + samples * (1 - mask_missing)

                # Decoder
                x_encoded_reconstructed = model.decoder(x_imputed)

                squared_error = (x_encoded_reconstructed - x_ground_truth) ** 2 # MSE error
                reconstruction_loss = torch.sqrt((squared_error * mask_true).sum() / mask_true.sum()) # RMSE error
                reconstruction_loss_epoch += reconstruction_loss.clone().detach().item()

                # Computation losses
                total_loss = self.hypers["alpha_weight"] * loss_mse + self.hypers["beta_weight"] * reconstruction_loss + p * self.hypers["gamma_weight"] * domain_loss
                # print(f"Loss mse: {loss_mse}, reconstruction {reconstruction_loss}, domain {domain_loss}")
                total_loss_epoch += total_loss.detach().cpu().numpy().mean()

                # Before first batch update
                # for name, param in model.encoder.named_parameters():
                #     if param.requires_grad:
                #         print(f"[Before] {name}: {param.data.view(-1)[0].item()}")

                if is_training:
                    optimizer.zero_grad()
                    total_loss.backward(retain_graph=True)

                    # for name, param in model.encoder.named_parameters():
                    #     print("ENC", name, "grad:", None if param.grad is None else param.grad.abs().mean().item())
                    # for name, param in model.decoder.named_parameters():
                    #     print("DEC", name, "grad:", None if param.grad is None else param.grad.abs().mean().item())
                    # for name, param in model.domain_classifier.named_parameters():
                    #     print("DOM", name, "grad:", None if param.grad is None else param.grad.abs().mean().item())

                    optimizer.step()

                # for name, param in model.encoder.named_parameters():
                #     if param.requires_grad:
                #         print(f"[After] {name}: {param.data.view(-1)[0].item()}")

            task_specific_loss = self.hypers["alpha_weight"] * (gain_loss_epoch/len(dataloader)) + self.hypers["beta_weight"] * (reconstruction_loss_epoch/len(dataloader))
            task_specific_loss = task_specific_loss.item()

        metrics.update(
            mode="Train" if is_training else "Test",
            loss=task_specific_loss,
            loss_gain=gain_loss_epoch/len(dataloader),
            loss_domain_classifier= domain_loss_epoch/len(dataloader),
            loss_model=total_loss_epoch/len(dataloader),
            rmse=reconstruction_loss_epoch/len(dataloader),
            domain_acc=domain_accuracy_epoch/len(dataloader)
        )

        return

    def train(self):

        n_folds = self.hypers["num_folds"]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.hypers["seed"])

        # Map project names to numeric domain labels
        projects = self.data.domain_labels.unique()
        project_to_number = {name: idx for idx, name in enumerate(projects)} # map each project with a number
        domain_labels = torch.tensor(self.data.domain_labels.map(project_to_number).to_numpy(), dtype=torch.long) # Needed for DANN training

        X_missing = torch.tensor(self.data.dataset_missing.values, dtype=torch.float32)
        X_true = torch.tensor(self.data.dataset_normalized.values, dtype=torch.float32)
        y = domain_labels

        metrics = MetricsTracker()
        evaluation = EvaluationTracker(
            projects=self.data.domain_labels,
            scaler=self.data.scaler,
            save_dir=self.save_dir
        )

        for fold, (train_index, test_index) in enumerate(skf.split(X_missing, y), 1):
            print(f"\n====== Fold {fold} ======")

            X_missing_train, X_missing_test = X_missing[train_index], X_missing[test_index]
            X_true_train, X_true_test = X_true[train_index], X_true[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # only needed for correlation purposes
            X_test_sample_names = self.data.dataset_missing.index[test_index].tolist()
            sample_to_project = self.data.sample_to_project

            train_dataset = TensorDataset(X_missing_train, y_train, X_true_train)
            test_dataset = TensorDataset(X_missing_test, y_test, X_true_test)

            # balance class distribution
            train_labels = torch.tensor([y for _, y, _ in train_dataset]) 
            class_samples_count = torch.bincount(train_labels)
            weights = 1. / class_samples_count
            sample_weights = weights[train_labels]

            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

            train_loader = DataLoader(train_dataset, batch_size=self.hypers["batch_size"], sampler=sampler)
            test_loader = DataLoader(test_dataset, batch_size=self.hypers["batch_size"])

            # Initialize Model and Optimizer
            model = self.initialize_model()
            optimizer, scheduler = self.initialize_optimizer_scheduler(model)

            # Train and Test Model
            num_epochs = self.hypers["num_epochs"]

            early_stopper = EarlyStopping(patience=self.early_stop_patience)

            for epoch in range(1, num_epochs+1):
                if epoch % 10 == 0 or epoch == 1:
                    print(f"\n--- [Fold {fold}] Epoch {epoch}/{num_epochs} ---")

                # Train
                # print("Train:")
                self.set_model_mode(model, is_training=True)

                # Stage A (pretrain)
                # if epoch < 100:
                #     self.hypers["gamma_weight"] = 0.0
                #     # run N_pre epochs
                # else:
                #     # Stage B (fine-tune)
                #     self.hypers["gamma_weight"] = 0.25
                #     # optionally use gamma warm-up below

                p = float(epoch) / num_epochs # training progress 0 -> 1
                self.epoch(model, train_loader, optimizer, p, metrics=metrics, is_training=True)

                # Test
                # print("\nTest:")
                self.set_model_mode(model, is_training=False)

                with torch.no_grad():
                    self.epoch(model, test_loader, optimizer, p, metrics=metrics, is_training=False)
                    loss_val = metrics.get_last_epoch_metric("loss_val")
                    rmse_val = metrics.get_last_epoch_metric("rmse_val")
                    scheduler.step(rmse_val)

                if epoch % 10 == 0 or epoch == 1: # todo delete just for benchmark purposes
                    metrics.print_metrics_epoch("Test")

                # Early Stopping
                if early_stopper.step(rmse_val, epoch):
                    warnings.warn(f"Early stopping at epoch {epoch}.")    
                    break     
 
            metrics.new_fold()

            # Fold's correlation
            X_test_hat, _ = model(X_missing_test)
            del model, optimizer, train_loader, test_loader, train_dataset, test_dataset
            torch.cuda.empty_cache()

            pearson_stats, rmse_stats = evaluation.evaluate_fold(
                X_true_test, X_test_hat, 
                X_test_sample_names, sample_to_project, 
            )

        pearson_stats_fold, rmse_stats_fold = evaluation.finalize()
        print("All folds stats")
        print(pearson_stats_fold)
        print(rmse_stats_fold)
        evaluation.plot()

        # Plots over folds
        metrics.mean_over_folds()
        metrics.plot_task_specific_losses(self.save_dir)
        metrics.plot_adversarial_losses(self.save_dir)
        metrics.plot_model_losses(self.save_dir)
        metrics.plot_domain_accuracies(self.save_dir)
        metrics.plot_rmses(self.save_dir)
        print(f"Train loss {metrics.avg_metrics['loss_train']}")
        print(f"Val loss {metrics.avg_metrics['loss_val']}")


        # ===== Final Training on Full Data =====

        print("\n🏁 Training final model...\n")

        train_dataset = TensorDataset(X_missing, y, X_true)
        train_loader = DataLoader(train_dataset, batch_size=self.hypers["batch_size"], sampler=sampler)
        
        final_model = self.initialize_model()
        final_optimizer, scheduler = self.initialize_optimizer_scheduler(final_model)
        self.set_model_mode(final_model, is_training=True)

        #todo como usar um scheduler no final model?

        final_metrics = MetricsTracker()
        early_stopper = EarlyStopping(self.early_stop_patience)

        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

            p = float(epoch) / num_epochs # training progress 0 -> 1
            self.epoch(final_model, train_loader, final_optimizer, p, metrics=final_metrics, is_training=True)
            # loss_train = final_metrics.get_last_epoch_metric("loss_train")

            # Early Stopping
            # todo ver se faz sentido ter um early stopper no train do modelo final
        print(f"Training time: {time.time() - start_time}")
        logger.info("\nFinal Model trained.\n")

        if self.save:
            save_model(final_model, self.save_dir)

        return metrics.avg_metrics["loss_val"] # mean of the validation losses across the folds for the optimization process