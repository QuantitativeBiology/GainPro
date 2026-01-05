from gainpro.hypers import Params
from gainpro.dataset import Data
from gainpro.output import Metrics
from gainpro import utils

import torch
from torch import nn
import numpy as np

from tqdm import tqdm
import psutil

from torchinfo import summary


class Network:
    def __init__(self, hypers: Params, net_G, net_D, metrics: Metrics):

        # for w in net_D.parameters():
        #    nn.init.normal_(w, 0, 0.02)
        # for w in net_G.parameters():
        #    nn.init.normal_(w, 0, 0.02)

        # for w in net_D.parameters():
        #    nn.init.xavier_normal_(w)
        # for w in net_G.parameters():
        #    nn.init.xavier_normal_(w)

        for name, param in net_D.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        for name, param in net_G.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        self.hypers = hypers
        self.net_G = net_G
        self.net_D = net_D
        self.metrics = metrics

        self.optimizer_D = torch.optim.Adam(net_D.parameters(), lr=hypers.lr_D)
        self.optimizer_G = torch.optim.Adam(net_G.parameters(), lr=hypers.lr_G)

        # print(summary(net_G))

    def generate_sample(self, data, mask):
        dim = data.shape[1]
        size = data.shape[0]

        Z = torch.rand((size, dim)) * 0.01
        missing_data_with_noise = mask * data + (1 - mask) * Z
        input_G = torch.cat((missing_data_with_noise, mask), 1).float()

        return self.net_G(input_G)

    def impute(self, data: Data):
        sample_G = self.generate_sample(data.dataset_scaled, data.mask)
        data_imputed_scaled = data.dataset_scaled * data.mask + sample_G * (
            1 - data.mask
        )
        self.metrics.data_imputed = data.scaler.inverse_transform(
            data_imputed_scaled.detach().numpy()
        )

        utils.create_csv(
            self.metrics.data_imputed,
            f"{self.hypers.output_folder}{self.hypers.output}",
            self.hypers.header,
        )

    def _evaluate_impute(self, data: Data):
        sample_G = self.generate_sample(data.ref_dataset_scaled, data.ref_mask)
        data_imputed_scaled = data.ref_dataset_scaled * data.ref_mask + sample_G * (
            1 - data.ref_mask
        )
        self.metrics.ref_data_imputed = data.scaler.inverse_transform(
            data_imputed_scaled.detach().numpy()
        )

        test_idx = torch.nonzero((data.mask - data.ref_mask) == 1)

        ref_imputed = np.empty((len(test_idx), 4))
        for i, id in enumerate(test_idx):
            ref_imputed[i] = np.array(
                [
                    data.dataset[tuple(id)],
                    self.metrics.ref_data_imputed[tuple(id)],
                    id[0],
                    id[1],
                ]
            )

        utils.create_csv(
            ref_imputed,
            f"{self.hypers.output_folder}test_imputed",
            ["original", "imputed", "sample", "feature"],
        )

    def _update_G(self, batch, mask, hint, Z, loss):
        loss_mse = nn.MSELoss(reduction="none")

        ones = torch.ones_like(batch)

        new_X = mask * batch + (1 - mask) * Z
        input_G = torch.cat((new_X, mask), 1).float()
        sample_G = self.net_G(input_G)
        fake_X = new_X * mask + sample_G * (1 - mask)

        fake_input_D = torch.cat((fake_X, hint), 1).float()
        fake_Y = self.net_D(fake_input_D)

        # print(batch, mask, ones.reshape(fake_Y.shape), fake_Y, loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask), (loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask)).mean())
        loss_G_entropy = (
            loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1 - mask)
        ).mean()
        loss_G_mse = (
            loss_mse((sample_G * mask).float(), (batch * mask).float())
        ).mean()

        loss_G = loss_G_entropy + self.hypers.alpha * loss_G_mse

        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        return loss_G

    def _update_D(self, batch, mask, hint, Z, loss):
        new_X = mask * batch + (1 - mask) * Z

        input_G = torch.cat((new_X, mask), 1).float()

        sample_G = self.net_G(input_G)
        fake_X = new_X * mask + sample_G * (1 - mask)
        fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
        fake_Y = self.net_D(fake_input_D)
        loss_D = (loss(fake_Y.float(), mask.float())).mean()
        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()

        return loss_D

    def train_ref(self, data: Data, missing_header):

        dim = data.dataset_scaled.shape[1]
        train_size = data.dataset_scaled.shape[0]

        if train_size < self.hypers.batch_size:
            self.hypers.batch_size = train_size
            print(
                "Batch size is larger than the number of samples\nReducing batch size to the number of samples\n"
            )

        # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        pbar = tqdm(range(self.hypers.num_iterations))
        for it in pbar:

            mb_idx = utils.sample_idx(train_size, self.hypers.batch_size)

            batch = data.dataset_scaled[mb_idx].detach().clone()
            mask_batch = data.mask[mb_idx].detach().clone()
            hint_batch = data.hint[mb_idx].detach().clone()
            ref_batch = data.ref_dataset_scaled[mb_idx].detach().clone()

            Z = torch.rand((self.hypers.batch_size, dim)) * 0.01
            self.metrics.loss_D[it] = self._update_D(
                batch, mask_batch, hint_batch, Z, loss
            )
            self.metrics.loss_G[it] = self._update_G(
                batch, mask_batch, hint_batch, Z, loss
            )

            sample_G = self.generate_sample(batch, mask_batch)

            self.metrics.loss_MSE_train[it] = (
                loss_mse(mask_batch * batch, mask_batch * sample_G)
            ).mean()

            self.metrics.loss_MSE_test[it] = (
                loss_mse((1 - mask_batch) * ref_batch, (1 - mask_batch) * sample_G)
            ).mean() / (1 - mask_batch).mean()

            if it % 100 == 0:
                s = f"{it}: loss D={self.metrics.loss_D[it]: .3f}  loss G={self.metrics.loss_G[it]: .3f}  rmse train={np.sqrt(self.metrics.loss_MSE_train[it]): .4f}  rmse test={np.sqrt(self.metrics.loss_MSE_test[it]): .3f}"
                pbar.clear()
                pbar.set_description(s)

            self.metrics.cpu[it] = psutil.cpu_percent()
            self.metrics.ram[it] = psutil.virtual_memory()[3] / 1000000000
            self.metrics.ram_percentage[it] = psutil.virtual_memory()[2]

        self.impute(data)

        if self.hypers.output_all == 1:
            utils.output(
                self.metrics.data_imputed,
                self.hypers.output_folder,
                self.hypers.output,
                missing_header,
                self.metrics.loss_D,
                self.metrics.loss_G,
                self.metrics.loss_MSE_train,
                self.metrics.loss_MSE_test,
                self.metrics.cpu,
                self.metrics.ram,
                self.metrics.ram_percentage,
                self.hypers.override,
            )

    def evaluate(self, data: Data, missing_header):

        dim = data.ref_dataset_scaled.shape[1]
        train_size = data.ref_dataset_scaled.shape[0]

        if train_size < self.hypers.batch_size:
            self.hypers.batch_size = train_size
            print(
                "\nBatch size is larger than the number of samples\nReducing batch size to the number of samples\n"
            )

        # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        pbar = tqdm(range(self.hypers.num_iterations))
        for it in pbar:
            mb_idx = utils.sample_idx(train_size, self.hypers.batch_size)

            train_batch = data.ref_dataset_scaled[mb_idx].detach().clone()
            train_mask_batch = data.ref_mask[mb_idx].detach().clone()
            train_hint_batch = data.ref_hint[mb_idx].detach().clone()
            test_batch = data.dataset_scaled[mb_idx].detach().clone()
            test_mask_batch = data.mask[mb_idx].detach().clone()
            Z = torch.rand((self.hypers.batch_size, dim)) * 0.01
            self.metrics.loss_D_evaluate[it] = self._update_D(
                train_batch, train_mask_batch, train_hint_batch, Z, loss
            )
            self.metrics.loss_G_evaluate[it] = self._update_G(
                train_batch, train_mask_batch, train_hint_batch, Z, loss
            )
            sample_G = self.generate_sample(train_batch, train_mask_batch)

            self.metrics.loss_MSE_train_evaluate[it] = (
                loss_mse(train_mask_batch * train_batch, train_mask_batch * sample_G)
            ).mean()

            self.metrics.loss_MSE_test[it] = (
                loss_mse(
                    (test_mask_batch - train_mask_batch) * test_batch,
                    (test_mask_batch - train_mask_batch) * sample_G,
                )
            ).mean() / (test_mask_batch - train_mask_batch).mean()

            if it % 100 == 0:
                s = f"{it}: loss D={self.metrics.loss_D_evaluate[it]: .3f}  loss G={self.metrics.loss_G_evaluate[it]: .3f}  rmse train={np.sqrt(self.metrics.loss_MSE_train_evaluate[it]): .4f}  rmse test={np.sqrt(self.metrics.loss_MSE_test[it]): .3f}"
                pbar.clear()
                pbar.set_description(s)

            self.metrics.cpu_evaluate[it] = psutil.cpu_percent()
            self.metrics.ram_evaluate[it] = psutil.virtual_memory()[3] / 1000000000
            self.metrics.ram_percentage_evaluate[it] = psutil.virtual_memory()[2]

        self._evaluate_impute(data)

        if self.hypers.output_all == 1:
            utils.output(
                self.metrics.ref_data_imputed,
                self.hypers.output_folder,
                self.hypers.output,
                missing_header,
                self.metrics.loss_D_evaluate,
                self.metrics.loss_G_evaluate,
                self.metrics.loss_MSE_train_evaluate,
                self.metrics.loss_MSE_test,
                self.metrics.cpu_evaluate,
                self.metrics.ram_evaluate,
                self.metrics.ram_percentage_evaluate,
                self.hypers.override,
            )

    def train(self, data: Data, missing_header):

        for name, param in self.net_D.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        for name, param in self.net_G.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
                # nn.init.uniform_(param)

        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.hypers.lr_D)
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=self.hypers.lr_G)

        dim = data.dataset_scaled.shape[1]
        train_size = data.dataset_scaled.shape[0]

        if train_size < self.hypers.batch_size:
            self.hypers.batch_size = train_size
            print(
                "\nBatch size is larger than the number of samples\nReducing batch size to the number of samples\n"
            )

        # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        loss = nn.BCELoss(reduction="none")
        loss_mse = nn.MSELoss(reduction="none")

        pbar = tqdm(range(self.hypers.num_iterations))
        for it in pbar:

            mb_idx = utils.sample_idx(train_size, self.hypers.batch_size)

            batch = data.dataset_scaled[mb_idx].detach().clone()
            mask_batch = data.mask[mb_idx].detach().clone()
            hint_batch = data.hint[mb_idx].detach().clone()

            Z = torch.rand((self.hypers.batch_size, dim)) * 0.01
            self.metrics.loss_D[it] = self._update_D(
                batch, mask_batch, hint_batch, Z, loss
            )
            self.metrics.loss_G[it] = self._update_G(
                batch, mask_batch, hint_batch, Z, loss
            )

            sample_G = self.generate_sample(batch, mask_batch)

            self.metrics.loss_MSE_train[it] = (
                loss_mse(mask_batch * batch, mask_batch * sample_G)
            ).mean()

            if it % 100 == 0:
                s = f"{it}: loss D={self.metrics.loss_D[it]: .3f}  loss G={self.metrics.loss_G[it]: .3f}  rmse train={np.sqrt(self.metrics.loss_MSE_train[it]): .4f}"
                pbar.clear()
                pbar.set_description(s)

            self.metrics.cpu[it] = psutil.cpu_percent()
            self.metrics.ram[it] = psutil.virtual_memory()[3] / 1000000000
            self.metrics.ram_percentage[it] = psutil.virtual_memory()[2]

        self.impute(data)

        if self.hypers.output_all == 1:
            utils.output(
                self.metrics.data_imputed,
                self.hypers.output_folder,
                self.hypers.output,
                missing_header,
                self.metrics.loss_D,
                self.metrics.loss_G,
                self.metrics.loss_MSE_train,
                self.metrics.loss_MSE_test,
                self.metrics.cpu,
                self.metrics.ram,
                self.metrics.ram_percentage,
                self.hypers.override,
            )
