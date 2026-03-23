import os
import errno
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold

from utils.paths import get_project_root

def kfold(
    dataset_path: Path,
    num_folds: int,
    seed: int=42,
) -> None:
    print("dataset path in generate dev split", dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_path)

    root_dir = get_project_root()
    save_dir = Path(f"{root_dir}/data/splits/{dataset_path.stem}/k-fold")
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, index_col=0)
    print("dataset shape in generate dev split", df.shape)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(df), start=1):
        print(f"fold id {fold_id}")
        print(f"train idx", trainval_idx)
        print(f"test idx", test_idx)
        fold_dir = save_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        np.save(f"{fold_dir}/trainval_idx.npy", trainval_idx)
        np.save(f"{fold_dir}/test_idx.npy", test_idx)

def trainval_test_split(
    dataset_path: Path,
    test_size: float,
    seed: int=42,
) -> None:
    if not dataset_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_path)

    name_dataset = dataset_path.parent.stem
    root_dir = get_project_root()
    save_dir = Path(f"{root_dir}/data/splits/{name_dataset}/{dataset_path.stem}")
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, index_col=0)
    num_samples = df.shape[0]
    # print("Number of samples", n_samples)
    indices = np.arange(num_samples)

    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    np.save(f"{save_dir}/trainval_idx.npy", trainval_idx)
    np.save(f"{save_dir}/test_idx.npy", test_idx)

def main(
    strategy: str,
    dataset_path: Path,
    num_folds: int=None,
    test_size: float=None,
    seed: int=42,
) -> None:
    
    if strategy == "train_test":
        trainval_test_split(
            dataset_path=dataset_path,
            test_size=test_size,
            seed=seed
        )
    elif strategy == "kfold":
        kfold(
            dataset_path=dataset_path,
            num_folds=num_folds,
            seed=seed,
        )
    else:
        raise SystemExit (f"Unknown {strategy} strategy. \n Strategies available are: train_test and kfold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["train_test", "kfold"],
        help="Dataset splitting strategy"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the directory containing the prepared dataset."
    )
    parser.add_argument(
        "--test-size", 
        type=float,
        default=None,
        help="Train/Test proportion (for train_test)"
    )
    parser.add_argument(
        "--num-folds", 
        type=int, 
        default=None,
        help="Number of folds (for kfold)"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=42,
        help="Seed"
    )
    args = parser.parse_args()
    main(
        strategy=args.strategy,
        dataset_path=args.dataset_path,
        num_folds=args.num_folds,
        test_size=args.test_size,
        seed=args.seed
    )