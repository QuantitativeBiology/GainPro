import torch
import psutil
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

import GenerativeProteomics.utils.helper_gain as helper_gain
from models.GainPro.hypers import Hypers
from models.GainPro.utils.dataset import Data
from GenerativeProteomics.models.GainPro.metrics import Metrics

#todo divide network vs. train


class GainPro(nn.Module):
    def __init__(
        cls,
        input_dim: int,
        # hypers: Hypers
    ) -> GainPro:
        super().__init__()

        cls.input_dim = input_dim
        cls.latent_dim = cls.input_dim

        # Generator
        cls.generator = nn.Sequential(
            nn.Linear(cls.latent_dim * 2, cls.latent_dim),
            nn.ReLU(),
            nn.Linear(cls.latent_dim, cls.latent_dim),
            nn.ReLU(),
            nn.Linear(cls.latent_dim, cls.latent_dim),
            nn.Sigmoid(),
        )

        # Discriminator
        cls.discriminator = nn.Sequential(
            nn.Linear(cls.latent_dim, cls.latent_dim),
            nn.ReLU(),
            nn.Linear(cls.latent_dim, cls.latent_dim),
            nn.ReLU(),
            nn.Linear(cls.latent_dim, cls.latent_dim),
            nn.Sigmoid(),
        )

        # cls.hypers = hypers
        # self.metrics = metrics

    def _init_weights(cls):
        for name, param in cls.generator.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

        for name, param in cls.discriminator.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def generate_sample(cls, data, mask):
        dim = data.shape[1]
        size = data.shape[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Z = torch.rand((size, dim), device=device) * 0.01

        # Update to run on GPU
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        mask = mask.to(device)

        missing_data_with_noise = mask * data.to(device) + (1 - mask) * Z
        input_G = torch.cat((missing_data_with_noise, mask), 1).float()

        return cls.net_G(input_G)

    def impute(cls, data: Data):
        sample_G = cls.generate_sample(data.dataset_scaled, data.observed_mask)
        device = sample_G.device
        data_imputed_scaled = data.dataset_scaled.to(device) * data.observed_mask.to(device) + sample_G * (
            1 - data.observed_mask.to(device)
        )

        cls.metrics.data_imputed = data.scaler.inverse_transform(
            data_imputed_scaled.detach().cpu().numpy()
        )

        helper_gain.create_csv(
            cls.metrics.data_imputed,
            f"{cls.hypers.output_folder}/{cls.hypers.output}",
            cls.hypers.header,
        )
    
    def _evaluate_impute(cls, data: Data):
        # Generate imputed sample
        sample_G = cls.generate_sample(data.ref_dataset_scaled, data.ref_mask)

        # Combine observed and imputed values
        data_imputed_scaled = data.ref_dataset_scaled * data.ref_mask + sample_G * (1 - data.ref_mask)

        # Move to CPU, detach, and convert to NumPy safely
        ref_data_imputed_np = data_imputed_scaled.detach().cpu().numpy()
        cls.metrics.ref_data_imputed = data.scaler.inverse_transform(ref_data_imputed_np)

        # Indices of the test/masked entries
        test_idx = torch.nonzero((data.observed_mask - data.ref_mask) == 1, as_tuple=False)

        n_features = data.dataset.shape[1]

        rows = []
        # Iterate over test indices
        for idx in test_idx:
            idx_tuple = tuple(idx.tolist())
            original_val = data.dataset[idx_tuple].detach().cpu().item()  # get scalar safely
            imputed_val = cls.metrics.ref_data_imputed[idx_tuple]
            mask_val = data.ref_mask[idx_tuple].item()

            # Append row for each feature
            for _ in range(n_features):
                rows.append([
                    data.samples[0],
                    data.columns[1],
                    original_val,
                    imputed_val,
                    mask_val,
                ])

        # Convert to DataFrame
        # ref_imputed = pd.DataFrame(rows, columns=["original", "imputed", "mask"])
        ref_imputed = pd.DataFrame(rows, columns=["sample", "protein", "original", "imputed", "mask"])

        # Save CSV
        helper_gain.create_csv(
            ref_imputed,
            f"{cls.hypers.output_folder}/{cls.hypers.output}_test_imputed",
            ["sample", "protein", "original", "imputed", "mask"],
        )

        # Explicit cleanup
        del sample_G, data_imputed_scaled, ref_data_imputed_np, rows
        torch.cuda.empty_cache()  # release any remaining GPU memory


    # def _evaluate_impute(cls, data: Data):
    #     sample_G = cls.generate_sample(data.ref_dataset_scaled, data.ref_mask)
    #     data_imputed_scaled = data.ref_dataset_scaled * data.ref_mask + sample_G * (
    #         1 - data.ref_mask
    #     )
    #     cls.metrics.ref_data_imputed = data.scaler.inverse_transform(
    #         data_imputed_scaled.detach().cpu().numpy()
    #     )

    #     test_idx = torch.nonzero((data.observed_mask - data.ref_mask) == 1)
    #     print(test_idx)

    #     # ref_imputed = np.empty((len(test_idx), 5))
    #     # for i, id in enumerate(test_idx):
    #     #     ref_imputed[i, 0] = data.samples[0]
    #     #     ref_imputed[i, 1] = data.columns[1]
    #     #     ref_imputed[i, 2] = data.dataset[tuple(id)].detach().cpu().numpy()
    #     #     ref_imputed[i, 3] = cls.metrics.ref_data_imputed[tuple(id)]
    #     #     ref_imputed[i, 4] = data.ref_mask[tuple(id)].item()
        
    #     n_features = data.dataset.shape[1]

    #     rows = []
    #     for _, idx in enumerate(test_idx):
    #         for _ in range(n_features):
    #             rows.append([
    #                 idx[0],
    #                 idx[1],
    #                 data.dataset[tuple(idx)].detach().cpu().numpy(),
    #                 cls.metrics.ref_data_imputed[tuple(idx)],
    #                 data.ref_mask[tuple(idx)].item(),
    #             ])
        
    #     ref_imputed = pd.DataFrame(rows)

    #     util.create_csv(
    #         ref_imputed,
    #         f"{cls.hypers.output_folder}/{cls.hypers.output}_test_imputed",
    #         # ["original", "imputed", "sample", "feature"],
    #         ["sample", "protein", "original", "imputed", "mask"],
    #     )

    def _update_G(cls, batch, mask, hint, Z, loss):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_mse = nn.MSELoss(reduction="none")

        ones = torch.ones_like(batch)

        batch = batch.to(device)

        new_X = mask * batch + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
        sample_G = cls.net_G(input_G)
        fake_X = new_X * mask + sample_G * (1 - mask)

        fake_input_D = torch.cat((fake_X, hint), 1).float()
        fake_Y = cls.net_D(fake_input_D)

        # print(batch, mask, ones.reshape(fake_Y.shape), fake_Y, loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask), (loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask)).mean())
        loss_G_entropy = (
            loss(fake_Y, ones.reshape(fake_Y.shape).float().to(device)) * (1 - mask)
        ).mean()
        loss_G_mse = (
            loss_mse((sample_G * mask).float(), (batch * mask).float())
        ).mean()

        loss_G = cls.hypers.alpha * loss_G_entropy + loss_G_mse

        cls.optimizer_G.zero_grad()
        loss_G.backward()
        cls.optimizer_G.step()

        return loss_G

    def _update_D(cls, batch, mask, hint, Z, loss):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Update to run on GPU
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device)

        batch = batch.to(device)

        new_X = mask * batch + (1 - mask) * Z

        input_G = torch.cat((new_X, mask), 1).float()
        input_G = input_G.to(device)

        sample_G = cls.net_G(input_G)
        fake_X = new_X * mask + sample_G * (1 - mask)
        fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
        fake_Y = cls.net_D(fake_input_D)
        loss_D = (loss(fake_Y.float(), mask.float())).mean()
        cls.optimizer_D.zero_grad()
        loss_D.backward()
        cls.optimizer_D.step()

        return loss_D
    
    def _compute_loss_G(cls, batch, mask, hint, Z, loss):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_mse = nn.MSELoss(reduction="none")

        ones = torch.ones_like(batch)

        batch = batch.to(device)

        new_X = mask * batch + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
        sample_G = cls.net_G(input_G)
        fake_X = new_X * mask + sample_G * (1 - mask)

        fake_input_D = torch.cat((fake_X, hint), 1).float()
        fake_Y = cls.net_D(fake_input_D)

        loss_G_entropy = (
            loss(fake_Y, ones.reshape(fake_Y.shape).float().to(device)) * (1 - mask)
        ).mean()
        loss_G_mse = (
            loss_mse((sample_G * mask).float(), (batch * mask).float())
        ).mean()

        loss_G = loss_G_entropy + cls.hypers.alpha * loss_G_mse
        return loss_G
    
    def _compute_loss_D(cls, batch, mask, hint, Z, loss):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Update to run on GPU
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device)

        batch = batch.to(device)

        new_X = mask * batch + (1 - mask) * Z

        input_G = torch.cat((new_X, mask), 1).float()
        input_G = input_G.to(device)

        sample_G = cls.net_G(input_G)
        fake_X = new_X * mask + sample_G * (1 - mask)
        fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
        fake_Y = cls.net_D(fake_input_D)

        loss_D = (loss(fake_Y.float(), mask.float())).mean()
        return loss_D

    def train_ref(cls, data: Data, missing_header):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim = data.dataset_scaled.shape[1]
        train_size = data.dataset_scaled.shape[0]

        if train_size < cls.hypers.batch_size:
            cls.hypers.batch_size = train_size
            print(
                "Batch size is larger than the number of samples\nReducing batch size to the number of samples\n"
            )

        # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        pbar = tqdm(range(cls.hypers.num_iterations))
        for it in pbar:

            mb_idx = helper_gain.sample_idx(train_size, cls.hypers.batch_size)

            batch = data.dataset_scaled[mb_idx].detach().clone().to(device)
            mask_batch = data.observed_mask[mb_idx].detach().clone().to(device)
            hint_batch = data.hint[mb_idx].detach().clone().to(device)
            ref_batch = data.ref_dataset_scaled[mb_idx].detach().clone().to(device)

            Z = torch.rand((cls.hypers.batch_size, dim), device=device) * 0.01
            cls.metrics.loss_D[it] = cls._update_D(
                batch, mask_batch, hint_batch, Z, loss
            )
            cls.metrics.loss_G[it] = cls._update_G(
                batch, mask_batch, hint_batch, Z, loss
            )

            sample_G = cls.generate_sample(batch, mask_batch)

            cls.metrics.loss_MSE_train[it] = (
                loss_mse(mask_batch * batch, mask_batch * sample_G)
            ).mean()

            cls.metrics.loss_MSE_test[it] = (
                loss_mse((1 - mask_batch) * ref_batch, (1 - mask_batch) * sample_G)
            ).mean() / (1 - mask_batch).mean()

            if it % 100 == 0:
                s = f"{it}: loss D={cls.metrics.loss_D[it]: .3f}  loss G={cls.metrics.loss_G[it]: .3f}  rmse train={np.sqrt(cls.metrics.loss_MSE_train[it]): .4f}  rmse test={np.sqrt(cls.metrics.loss_MSE_test[it]): .3f}"
                print(s)
                pbar.clear()
                pbar.set_description(s)


            cls.metrics.cpu[it] = psutil.cpu_percent()
            cls.metrics.ram[it] = psutil.virtual_memory()[3] / 1000000000
            cls.metrics.ram_percentage[it] = psutil.virtual_memory()[2]

        cls.impute(data)

        if cls.hypers.output_all == 1:
            helper_gain.output(
                cls.metrics.data_imputed,
                cls.hypers.output_folder,
                cls.hypers.output,
                missing_header,
                cls.metrics.loss_D,
                cls.metrics.loss_G,
                cls.metrics.loss_MSE_train,
                cls.metrics.loss_MSE_test,
                cls.metrics.cpu,
                cls.metrics.ram,
                cls.metrics.ram_percentage,
                cls.hypers.override,
            )

    def _evaluate_impute_cv(
        cls, 
        data: Data,
        test_idx,
    ) -> None:
        ref_dataset_scaled_test = data.ref_dataset_scaled[test_idx]
        ref_mask_test = data.ref_mask[test_idx]
        ref_dataset_test = data.ref_dataset[test_idx]
        dataset_test = data.dataset[test_idx]
        mask_test = data.observed_mask[test_idx]

        sample_G = cls.generate_sample(ref_dataset_scaled_test, ref_mask_test)
        data_imputed_scaled = ref_dataset_scaled_test * ref_mask_test + sample_G * (
            1 - ref_mask_test
        )
        cls.metrics.ref_data_imputed = data.scaler.inverse_transform(
            data_imputed_scaled.detach().cpu().numpy()
        )

        rows = []
        n_features = ref_dataset_test.shape[1]

        for idx in range(dataset_test.shape[0]):
            for feature_id in range(n_features):
                rows.append([
                    data.samples[idx],
                    data.columns[feature_id],
                    dataset_test[idx, feature_id].detach().cpu().numpy(),
                    cls.metrics.ref_data_imputed[idx, feature_id],
                    mask_test[idx, feature_id].item(),
                ])

        ref_imputed = pd.DataFrame(rows)

        helper_gain.create_csv(
            ref_imputed,
            f"{cls.hypers.output_folder}/{cls.hypers.output}_test_imputed",
            ["sample", "protein", "original", "imputed", "mask"],
        )

    def evaluate_cv(
        cls,
        data,
        input_dim,
        train_idx,
        test_idx,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        # train on train set
        pbar = tqdm(range(cls.hypers.num_iterations))
        for it in pbar:
            reference_train = data.ref_dataset_scaled[train_idx].detach().clone()
            mask_train = data.ref_mask[train_idx].detach().clone()
            hint_train = data.ref_hint[train_idx].detach().clone()

            reference_test = data.ref_dataset_scaled[test_idx].detach().clone()
            missing_test = data.dataset_scaled[test_idx].detach().clone()
            mask_test = data.observed_mask[test_idx].detach().clone()

            # print("batch size", cls.hypers.batch_size)
            # print("input dim", input_dim)

            Z = torch.rand((reference_train.shape[0], input_dim), device=device) * 0.01
            # print("z shape", Z.shape)

            cls.metrics.loss_D_evaluate[it] = cls._update_D(
                reference_train, mask_train, hint_train, Z, loss
            )
            cls.metrics.loss_G_evaluate[it] = cls._update_G(
                reference_train, mask_train, hint_train, Z, loss
            )
            sample_G = cls.generate_sample(reference_train, mask_train)

            cls.metrics.loss_MSE_train_evaluate[it] = (
                loss_mse(mask_train * reference_train, mask_train * sample_G)
            ).mean()

            sample_G_test = cls.generate_sample(reference_test, mask_test)
            cls.metrics.loss_MSE_test[it] = (
                loss_mse(
                    (reference_test - missing_test) * missing_test,
                    (mask_test) * sample_G_test,
                )
            ).mean() / mask_test.mean()

            if it % 100 == 0:
                s = f"{it}: loss D={cls.metrics.loss_D_evaluate[it]: .3f}  loss G={cls.metrics.loss_G_evaluate[it]: .3f}  rmse train={np.sqrt(cls.metrics.loss_MSE_train_evaluate[it]): .4f}  rmse test={np.sqrt(cls.metrics.loss_MSE_test[it]): .3f}"
                print(s)
                pbar.clear()
                pbar.set_description(s)

        # test_dataset = data.dataset_scaled[test_idx].cpu().numpy()
        # print("type test dataset", type((test_dataset)))
        # print("device test dataset", test_dataset.device)
        # print("test dataset", test_dataset)
        # test_data = Data(
        #     dataset=data.dataset_scaled[test_idx].cpu().numpy(),
        #     ref=data.ref_dataset_scaled[test_idx],
        #     samples=data.samples,
        #     columns=data.columns,
        #     miss_rate=cls.hypers.miss_rate,
        #     hint_rate=cls.hypers.hint_rate,
        # )
        cls._evaluate_impute_cv(data, test_idx)


    def evaluate(cls, data: Data, missing_header):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim = data.ref_dataset_scaled.shape[1]
        train_size = data.ref_dataset_scaled.shape[0]

        if train_size < cls.hypers.batch_size:
            cls.hypers.batch_size = train_size
            print(
                "\nBatch size is larger than the number of samples\nReducing batch size to the number of samples\n"
            )

        # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        pbar = tqdm(range(cls.hypers.num_iterations))
        for it in pbar:
        # for it in range(cls.hypers.num_iterations):
            mb_idx = helper_gain.sample_idx(train_size, cls.hypers.batch_size)

            train_batch = data.ref_dataset_scaled[mb_idx].detach().clone()
            train_mask_batch = data.ref_mask[mb_idx].detach().clone()
            train_hint_batch = data.ref_hint[mb_idx].detach().clone()
            test_batch = data.dataset_scaled[mb_idx].detach().clone()
            test_mask_batch = data.observed_mask[mb_idx].detach().clone()
            Z = torch.rand((cls.hypers.batch_size, dim), device=device) * 0.01

            cls.metrics.loss_D_evaluate[it] = cls._update_D(
                train_batch, train_mask_batch, train_hint_batch, Z, loss
            )
            cls.metrics.loss_G_evaluate[it] = cls._update_G(
                train_batch, train_mask_batch, train_hint_batch, Z, loss
            )
            sample_G = cls.generate_sample(train_batch, train_mask_batch)

            cls.metrics.loss_MSE_train_evaluate[it] = (
                loss_mse(train_mask_batch * train_batch, train_mask_batch * sample_G)
            ).mean()

            cls.metrics.loss_MSE_test[it] = (
                loss_mse(
                    (test_mask_batch - train_mask_batch) * test_batch,
                    (test_mask_batch - train_mask_batch) * sample_G,
                )
            ).mean() / (test_mask_batch - train_mask_batch).mean()

            if it % 100 == 0:
                s = f"{it}: loss D={cls.metrics.loss_D_evaluate[it]: .3f}  loss G={cls.metrics.loss_G_evaluate[it]: .3f}  rmse train={np.sqrt(cls.metrics.loss_MSE_train_evaluate[it]): .4f}  rmse test={np.sqrt(cls.metrics.loss_MSE_test[it]): .3f}"
                print(s)
                pbar.clear()
                pbar.set_description(s)

            cls.metrics.cpu_evaluate[it] = psutil.cpu_percent()
            cls.metrics.ram_evaluate[it] = psutil.virtual_memory()[3] / 1000000000
            cls.metrics.ram_percentage_evaluate[it] = psutil.virtual_memory()[2]

        cls._evaluate_impute(data)

        if cls.hypers.output_all == 1:
            helper_gain.output(
                cls.metrics.ref_data_imputed,
                cls.hypers.output_folder,
                cls.hypers.output,
                missing_header,
                cls.metrics.loss_D_evaluate,
                cls.metrics.loss_G_evaluate,
                cls.metrics.loss_MSE_train_evaluate,
                cls.metrics.loss_MSE_test,
                cls.metrics.cpu_evaluate,
                cls.metrics.ram_evaluate,
                cls.metrics.ram_percentage_evaluate,
                cls.hypers.override,
            )

    def fit(cls, data: Data, missing_header):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for name, param in cls.net_D.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        for name, param in cls.net_G.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        cls.optimizer_D = torch.optim.Adam(cls.net_D.parameters(), lr=cls.hypers.lr_D)
        cls.optimizer_G = torch.optim.Adam(cls.net_G.parameters(), lr=cls.hypers.lr_G)

        dim = data.dataset_scaled.shape[1]
        train_size = data.dataset_scaled.shape[0]

        if train_size < cls.hypers.batch_size:
            cls.hypers.batch_size = train_size
            print(
                "\nBatch size is larger than the number of samples\nReducing batch size to the number of samples\n"
            )

        # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        pbar = tqdm(range(cls.hypers.num_iterations))
        for it in pbar:

            mb_idx = helper_gain.sample_idx(train_size, cls.hypers.batch_size)

            batch = data.dataset_scaled[mb_idx].detach().clone()
            mask_batch = data.observed_mask[mb_idx].detach().clone()
            hint_batch = data.hint[mb_idx].detach().clone()

            Z = torch.rand((cls.hypers.batch_size, dim), device=device) * 0.01
            cls.metrics.loss_D[it] = cls._update_D(
                batch, mask_batch, hint_batch, Z, loss
            )
            cls.metrics.loss_G[it] = cls._update_G(
                batch, mask_batch, hint_batch, Z, loss
            )

            sample_G = cls.generate_sample(batch, mask_batch)

            cls.metrics.loss_MSE_train[it] = (
                loss_mse(mask_batch * batch, mask_batch * sample_G)
            ).mean()

            if it % 100 == 0:
                s = f"{it}: loss D={cls.metrics.loss_D[it]: .3f}  loss G={cls.metrics.loss_G[it]: .3f}  rmse train={np.sqrt(cls.metrics.loss_MSE_train[it]): .4f}"
                print(s)
                pbar.clear()
                pbar.set_description(s)

            cls.metrics.cpu[it] = psutil.cpu_percent()
            cls.metrics.ram[it] = psutil.virtual_memory()[3] / 1000000000
            cls.metrics.ram_percentage[it] = psutil.virtual_memory()[2]

        cls.impute(data)

        if cls.hypers.output_all == 1:
            helper_gain.output(
                cls.metrics.data_imputed,
                cls.hypers.output_folder,
                cls.hypers.output,
                missing_header,
                cls.metrics.loss_D,
                cls.metrics.loss_G,
                cls.metrics.loss_MSE_train,
                cls.metrics.loss_MSE_test,
                cls.metrics.cpu,
                cls.metrics.ram,
                cls.metrics.ram_percentage,
                cls.hypers.override,
            )