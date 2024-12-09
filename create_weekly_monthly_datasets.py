import numpy
import pandas as pd
import sys
import os

directory = "data/EcmP_stock_L_2016_24_mix"
target = "data/EcmP_stock_L_2016_24_mix_wm"

for f in os.listdir(directory):
    df = pd.read_csv(os.path.join(directory,f))
    for col in df.columns[1:]:
        if col!='close':
            df[f"{col}_week"] = df[col].rolling(window=5).mean()
            df[f"{col}_month"] = df[col].rolling(window=22).mean()
    df.iloc[21:,].to_csv(f"{os.path.join(target,f)[:-4]}_wm.csv", index=False)
