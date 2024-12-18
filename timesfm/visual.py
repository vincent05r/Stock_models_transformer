import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("combined.csv")
df_forecast = pd.read_csv("zero_shot_prediction.csv")

combined = df.join(df_forecast, how="inner", lsuffix="l_")#, on=["unique_id", "ds"])
#combined["ds"]

print(combined)

for uid in combined["unique_id"].unique():
    df_temp = combined[combined["unique_id"] == uid]
    plt.plot(df_temp["ds"],df_temp["close"],label="true")
    plt.plot(df_temp["ds"],df_temp["timesfm"],label="pred")
    plt.legend()
    plt.savefig("visualisations/" + uid + ".png")
    plt.clf()
