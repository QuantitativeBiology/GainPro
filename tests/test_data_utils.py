import pandas as pd
import numpy as np

from utils.data.data_utils import Data

def test_load_matrices():
    print("")

    reference = pd.DataFrame({
        "ProteinA": [np.nan, 2.0],
        "ProteinB": [3.0, 4.0],
        },
        index=["Sample1", "Sample2"]
    ).astype(np.float32)

    missing = pd.DataFrame({
        "ProteinA": [np.nan, 2.0],
        "ProteinB": [np.nan, 4.0],
        },
        index=["Sample1", "Sample2"]
    ).astype(np.float32)

    mask = ~reference.isna()
    domain = pd.DataFrame({
        "Domain": [1, 2]
        },
        index=["Sample1", "Sample2"]
    )

    print(f"Reference \n{reference}\n")
    print(f"Missing \n{missing}\n")
    print(f"Mask \n{mask}\n")
    print(f"Domain \n{domain}\n")

    data = Data(reference, missing, mask, domain)

    assert data.reference.equals(reference)
    assert data.missing.equals(missing)
    assert data.mask.equals(mask)
    assert data.domain.equals(domain)

def test_number_domains():
    print("")

    reference = pd.DataFrame({
        "ProteinA": [np.nan, 2.0],
        "ProteinB": [3.0, 4.0],
        },
        index=["Sample1", "Sample2"]
    ).astype(np.float32)

    missing = pd.DataFrame({
        "ProteinA": [np.nan, 2.0],
        "ProteinB": [np.nan, 4.0],
        },
        index=["Sample1", "Sample2"]
    ).astype(np.float32)

    mask = ~reference.isna()
    domain = pd.DataFrame({
        "Domain": [1, 2]
        },
        index=["Sample1", "Sample2"]
    )

    data = Data(reference, missing, mask, domain)
    assert data.n_domains == 2

def test_get_samples_names():
    print("")

    reference = pd.DataFrame({
        "ProteinA": [np.nan, 2.0],
        "ProteinB": [3.0, 4.0],
        },
        index=["Sample1", "Sample2"]
    ).astype(np.float32)

    missing = pd.DataFrame({
        "ProteinA": [np.nan, 2.0],
        "ProteinB": [np.nan, 4.0],
        },
        index=["Sample1", "Sample2"]
    ).astype(np.float32)

    mask = ~reference.isna()
    domain = pd.DataFrame({
        "Domain": [1, 2]
        },
        index=["Sample1", "Sample2"]
    )

    data = Data(reference, missing, mask, domain)
    assert data.get_samples_names() == ["Sample1", "Sample2"]

def test_get_sample_to_domain():
    print("")

    reference = pd.DataFrame({
        "ProteinA": [np.nan, 2.0],
        "ProteinB": [3.0, 4.0],
        },
        index=["Sample1", "Sample2"]
    ).astype(np.float32)

    missing = pd.DataFrame({
        "ProteinA": [np.nan, 2.0],
        "ProteinB": [np.nan, 4.0],
        },
        index=["Sample1", "Sample2"]
    ).astype(np.float32)

    mask = ~reference.isna()
    domain = pd.DataFrame({
        "Domain": [1, 2]
        },
        index=["Sample1", "Sample2"]
    )

    data = Data(reference, missing, mask, domain)
    assert data._get_sample_to_domain() == {"Sample1":1, "Sample2": 2}