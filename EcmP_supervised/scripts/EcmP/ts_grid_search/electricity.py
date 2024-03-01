import os 
import subprocess
from pathlib import Path

#Directory 
result_dir = "./logs/" + "EcmP_gridsearch"
log_dirs = ["./logs", result_dir]
for dir_path in log_dirs:
    os.makedirs(dir_path, exist_ok=True)

#logistics
model_name = "EcmP"
result_log_path = "./result_log/EcmP/electricity.txt"

root_path_name = "./data/ts_benchmark"
data_path_name = "electricity.csv"
model_id_name = "Electricity"
data_name = "custom"

#dataset components
