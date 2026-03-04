import pandas as pd
import numpy as np
import utils

df = pd.read_csv(
    "/home/leandrosobral/LeandroSobralThesis/ProtoGain/breast/results/lossD.csv"
)

print(df)


utils.create_output(
    df,
    "/home/leandrosobral/LeandroSobralThesis/ProtoGain/breast/results/lossD.csv",
    1,
)
