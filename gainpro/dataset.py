import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class Data:
    def __init__(self, dataset, miss_rate, hint_rate, ref=None):

        mask = np.where(np.isnan(dataset), 0.0, 1.0)
        dataset = np.where(mask, dataset, 0.0)

        range_scaler = (0, 1)
        self.scaler = MinMaxScaler(feature_range=range_scaler)
        dataset_scaled = self.scaler.fit_transform(dataset)

        dataset_scaled = np.where(mask, dataset_scaled, 0.0)
        hint = generate_hint(mask, hint_rate)

        self.dataset = torch.from_numpy(dataset)
        self.mask = torch.from_numpy(mask)
        self.hint = torch.from_numpy(hint)
        self.dataset_scaled = torch.from_numpy(dataset_scaled)

        if ref is not None:
            ref_mask = np.where(np.isnan(ref), 0.0, 1.0)
            ref_dataset = np.where(ref_mask, ref, 0.0)
            ref_hint = generate_hint(ref_mask, hint_rate)
            ref_dataset_scaled = self.scaler.transform(ref_dataset)

            self.ref_dataset = torch.from_numpy(ref_dataset)
            self.ref_mask = torch.from_numpy(ref_mask)
            self.ref_hint = torch.from_numpy(ref_hint)
            self.ref_dataset_scaled = torch.from_numpy(ref_dataset_scaled)
        else:
            self._create_ref(miss_rate, hint_rate)

        print("\nNumber of samples:", self.dataset.shape[0])
        print("Number of features:", self.dataset.shape[1])
        print("Missing Rate (%):", (1.0 - self.mask.mean().item()) * 100.0, "\n")

    def _create_ref(self, miss_rate, hint_rate):

        self.ref_mask = self.mask.detach().clone()
        self.ref_dataset = self.dataset.detach().clone()
        zero_idxs = torch.nonzero(self.mask == 1)
        chance = torch.rand(len(zero_idxs))
        miss = chance > miss_rate

        selected_idx = zero_idxs[~miss]
        for idx in selected_idx:
            self.ref_mask[tuple(idx)] = 0
            self.ref_dataset[tuple(idx)] = 0

        self.ref_hint = generate_hint(self.ref_mask, hint_rate)
        self.ref_dataset_scaled = torch.from_numpy(self.scaler.transform(self.ref_dataset))


def generate_hint(mask, hint_rate):
    hint_mask = generate_mask(mask, 1 - hint_rate)
    hint = mask * hint_mask

    return hint


def generate_mask(data, miss_rate):
    dim = data.shape[1]
    size = data.shape[0]
    A = np.random.uniform(0.0, 1.0, size=(size, dim))
    B = A > miss_rate
    mask = 1.0 * B

    return mask
