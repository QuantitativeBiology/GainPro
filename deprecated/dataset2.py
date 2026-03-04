import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


class Data:
    def __init__(
        self, 
        dataset, 
        miss_rate: float, 
        hint_rate: float,
        domain = None,
        ref = None
    ) -> "Data":

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.domain = torch.from_numpy(domain).to(device)

        mask = np.where(np.isnan(dataset), 0.0, 1.0)
        dataset = np.where(mask, dataset, 0.0)
        
        self.scaler = StandardScaler()
        dataset_scaled = self.scaler.fit_transform(dataset)

        dataset_scaled = np.where(mask, dataset_scaled, 0.0)
        hint = generate_hint(mask, hint_rate)

        self.dataset = torch.from_numpy(dataset).to(device)
        self.mask = torch.from_numpy(mask).to(device)
        # self.hint = torch.from_numpy(hint).to(device)
        self.hint = hint
        self.dataset_scaled = torch.from_numpy(dataset_scaled).to(device)

        if ref is not None:
            ref_mask = np.where(np.isnan(ref), 0.0, 1.0)
            ref_dataset = np.where(ref_mask, ref, 0.0)
            ref_hint = generate_hint(ref_mask, hint_rate)
            ref_dataset_scaled = self.scaler.transform(ref_dataset)

            self.ref_dataset = torch.from_numpy(ref_dataset).to(device)
            self.ref_mask = torch.from_numpy(ref_mask).to(device)
            self.ref_hint = torch.from_numpy(ref_hint).to(device)
            self.ref_dataset_scaled = torch.from_numpy(ref_dataset_scaled).to(device)
        else:
            self._create_ref(miss_rate, hint_rate) # original
            # self._read_ref_benchmark(dataset_missing, hint_rate)

        print("\nNumber of samples:", self.dataset.shape[0])
        print("Number of features:", self.dataset.shape[1])
        print("Missing Rate (%):", (1.0 - self.mask.mean().item()) * 100.0, "\n")
        print("Missing Rate (%):", (1.0 - self.ref_mask.mean().item()) * 100.0, "\n")

    def _read_ref_benchmark(cls, dataset_missing, hint_rate):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ref = dataset_missing

        ref_mask = np.where(np.isnan(ref), 0.0, 1.0)
        ref_dataset = np.where(ref_mask, ref, 0.0)
        ref_hint = generate_hint(ref_mask, hint_rate)
        ref_dataset_scaled = cls.scaler.transform(ref_dataset)

        # cls.dataset_scaled[cls.dataset_scaled==0] = np.nan
        # mask = np.where(np.isnan(cls.dataset_scaled.numpy()), 0.0, 1.0)
        # print((mask == ref_mask).all())

        cls.ref_dataset = torch.from_numpy(ref_dataset).to(device)
        cls.ref_mask = torch.from_numpy(ref_mask).to(device)
        cls.ref_hint = ref_hint
        # cls.ref_hint = torch.from_numpy(ref_hint)
        cls.ref_dataset_scaled = torch.from_numpy(ref_dataset_scaled).to(device)
        
        # print(f"Entries to mask: \n {pd.DataFrame(cls.ref_mask.numpy())}")
        # print(f"Original matrix: \n {pd.DataFrame(cls.dataset_scaled.numpy())}")
        # print(f"Masked matrix: \n {pd.DataFrame(cls.ref_dataset_scaled.numpy())}")

    def _create_ref(cls, miss_rate, hint_rate):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        cls.ref_mask = cls.mask.detach().clone().to(device)
        cls.ref_dataset = cls.dataset.detach().clone().to(device)

        size = cls.ref_dataset.shape[0] * cls.ref_dataset.shape[1]

        total_missing = torch.sum(cls.ref_dataset == 0).item()
        current_rate = total_missing / size

        print("Current rate", current_rate)

        # Target missingness
        target_rate = min(current_rate + miss_rate, 1.0)
        target_missing = int(target_rate * size)

        print("Target rate", target_rate)

        # How many more values to mask
        additional_to_mask = target_missing - total_missing
        if additional_to_mask <= 0:
            print("⚠️ No additional missingness needed (already too many NaNs).")

        # indices of currently observed and non-NaN (entries equal to 0) entries (only these can be newly masked)
        observed_idxs = torch.nonzero((cls.ref_mask == 1) & (cls.ref_dataset != 0))
        observed_count = observed_idxs.size(0)

        print("Eligible positions", observed_count)

        if additional_to_mask > observed_count:
            raise ValueError("Not enough eligible entries to mask the requested amount.")

        # Randomly sample indices to mask
        chosen = observed_idxs[torch.randperm(observed_count)[:additional_to_mask]]

        # Apply masking
        for i, j in chosen:
            cls.ref_dataset[i, j] = 0    # mark as missing
            cls.ref_mask[i, j] = 0       # update mask too

        # generate hint from the new mask
        cls.ref_hint = generate_hint(cls.ref_mask, hint_rate)
        cls.ref_dataset_scaled = torch.from_numpy(cls.scaler.transform(cls.ref_dataset.cpu().numpy())).to(device)       

    # def _create_ref(cls, miss_rate, hint_rate): # original create ref

    #     cls.ref_mask = cls.mask.detach().clone()
    #     cls.ref_dataset = cls.dataset.detach().clone()
    #     zero_idxs = torch.nonzero(cls.mask == 1)
    #     torch.manual_seed(1)
    #     chance = torch.rand(len(zero_idxs))
    #     miss = chance > miss_rate

    #     selected_idx = zero_idxs[~miss]
    #     for idx in selected_idx:
    #         cls.ref_mask[tuple(idx)] = 0
    #         cls.ref_dataset[tuple(idx)] = 0

    #     cls.ref_hint = generate_hint(cls.ref_mask, hint_rate)
    #     cls.ref_dataset_scaled = torch.from_numpy(cls.scaler.transform(cls.ref_dataset))


# def generate_hint(mask, hint_rate):
#     hint_mask = generate_mask(mask, 1 - hint_rate)
#     hint = mask * hint_mask

#     return hint

def generate_hint(mask, hint_rate): # adaptado para correr com o nosso modelo
    # version for DANN & GAIN hybrid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update to run on GPU
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
        mask = mask.to(device)


    hint_mask = generate_mask(mask, 1 - hint_rate)
    if isinstance(hint_mask, np.ndarray):
        hint_mask = torch.from_numpy(hint_mask).to(device)
        hint_mask = hint_mask.to(device)

    hint = mask * hint_mask

    return hint


def generate_mask(data, miss_rate):
    dim = data.shape[1]
    size = data.shape[0]
    A = np.random.uniform(0.0, 1.0, size=(size, dim))
    B = A > miss_rate
    mask = 1.0 * B

    return mask
