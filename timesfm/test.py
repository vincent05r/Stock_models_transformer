import timesfm
import pandas as pd
import os

# For Torch
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
  )


#help(tfm.forecast)

#import numpy as np
#forecast_input = [
#    np.sin(np.linspace(0, 20, 100)),
#    np.sin(np.linspace(0, 20, 200)),
#    np.sin(np.linspace(0, 20, 400)),
#]
#frequency_input = [0, 1, 2]
#
#point_forecast, experimental_quantile_forecast = tfm.forecast(
#    forecast_input,
#    freq=frequency_input,
#)

df = pd.read_csv("../data/EcmP_stock_L_2016_24/AAPL.csv")
df["ds"] = pd.to_datetime(df["date"])
df["unique_id"] = "AAPL"

for f in os.listdir("../data/EcmP_stock_L_2016_24"):
    df_new = pd.read_csv(os.path.join("../data/EcmP_stock_L_2016_24", f))
    df_new["ds"] = pd.to_datetime(df_new["date"])
    df_new["unique_id"] = f[:-4]
    df = pd.concat([df, df_new])


forecast_df = tfm.forecast_on_df(
    inputs=df,
    freq="D",  
    value_name="close",
    num_jobs=-1
)



print(df)
df.to_csv("combined.csv", index=False)
print(forecast_df)
print(forecast_df.shape)
print(forecast_df.dtypes)
forecast_df.to_csv("zero_shot_prediction.csv", index=False)

