import argparse
import errno
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2

def main(dir: str):
    dir = Path(dir)
    
    if not dir.is_dir():
        raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), dir)

    name_dataset = dir.parent.stem
    save_dir = Path(f"../data/splits/{name_dataset}")
    save_dir.mkdir(parents=True, exist_ok=True)

    reference_df = pd.read_csv(f"{dir}/reference.csv", index_col=0)
    n_samples = reference_df.shape[0]
    # print("Number of samples", n_samples)
    indices = np.arange(n_samples)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )

    np.save(f"{save_dir}/train_idx.npy", train_idx)
    np.save(f"{save_dir}/val_idx.npy", val_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dir", 
        type=str, 
        help="Path to directory with the prepared datasets"
    )

    args = parser.parse_args()

    main(args.dir)