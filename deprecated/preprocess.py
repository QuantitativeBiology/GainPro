from pathlib import Path
import pandas as pd

from helper import load_tsv_with_condition

X, sample_condition = load_tsv_with_condition(Path("raw/PXD030304.tsv"))

print(X.shape)

samples = sample_condition[sample_condition != "HEK293T"].index
X_samples = X.loc[samples]

print(X_samples.shape)

X_samples.to_csv("processed/PXD030304.csv", index=True)

df = pd.read_csv("processed/PXD030304.csv", index_col=0)
print(df.shape)