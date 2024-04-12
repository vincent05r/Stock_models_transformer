import os 
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd


#result grid
result_list = []
result_grid_path = "./result_log/EcmP/hsi_result_grid.txt"


#Directory 
result_dir = "./logs/" + "EcmP_gridsearch"
log_dirs = ["./logs", result_dir]
for dir_path in log_dirs:
    os.makedirs(dir_path, exist_ok=True)

#logistics
model_name = "EcmP"
result_log_path = "./result_log/EcmP/result_hsi.txt"



#dataset components
target = "close"
scale = str(1)
label_len = 2

root_path_name = "./data/stock_benchmark/"
data_path_name = "hsi_index_pct_spec_s.csv"
model_id_name = "hsi_index_pct_spec_s"
data_name = "stock_custom"

features: str = "MS"
enc_in = 9


#model components
dropout = 0.2
fc_dropout = 0.2
head_dropout = 0


#training components
decomposition = 1
des = 'Exp' 
train_epochs = 100
patience = 10
lradj = 'TST'
pct_start = 0.2
itr = 1
batch_size = 32
learning_rate = 0.0001

#patching setting
patch_len = 3
stride = 1


#Parameter grid
seq_lens = [7, 10, 15, 30, 50, 100]
pred_lens = [1]

e_layers_list = [2, 3, 4, 5]
n_heads_list = [4, 8]


model_size_multiplier = 4

model_var_preset = [

    {
        "d_patch" : 32,
        "d_model" : 64,
        "d_ff" : 64
    }

    ,

    {
        "d_patch" : 32,
        "d_model" : 64,
        "d_ff" : 128
    }

]

# patching_size_multiplier = 3

# patching_var_preset = [

#     {
#         "patch_len" : 8,
#         "stride" : 4
#     }

# ]



#
for random_seed in [2023]: #np.random.choice(3000, 5):
    for seq_len in seq_lens:
        for pred_len in pred_lens:
            for e_layers in e_layers_list:
                for n_heads in n_heads_list:
                    for msm_i in range(1, model_size_multiplier):
                        for mvp in model_var_preset:
                            d_patch = msm_i * mvp["d_patch"]
                            d_model = msm_i * mvp["d_model"]
                            d_ff = msm_i * mvp["d_ff"]
                            



                            log_file_path = f"{result_dir}/{model_name}_{model_id_name}_{seq_len}_{pred_len}_{learning_rate}_{batch_size}.log"

                            command = [
                                "python", "EcmP_supervised/run_longExp.py",
                                    "--target", target,
                                    "--scale", scale,
                                    "--decomposition", str(decomposition),
                                    "--result_log_path", str(result_log_path),
                                    "--random_seed", str(random_seed),
                                    "--is_training", str(1),
                                    "--root_path", root_path_name,
                                    "--data_path", data_path_name,
                                    "--model_id", f"{model_id_name}_{seq_len}_{pred_len}",
                                    "--model", model_name,
                                    "--data", data_name,
                                    "--features", features,
                                    "--seq_len", str(seq_len),
                                    "--label_len", str(label_len),
                                    "--pred_len", str(pred_len),
                                    "--enc_in", str(enc_in),
                                    "--e_layers", str(e_layers),
                                    "--n_heads", str(n_heads),
                                    "--d_patch", str(d_patch),
                                    "--d_model", str(d_model),
                                    "--d_ff", str(d_ff),
                                    "--dropout", str(dropout),
                                    "--fc_dropout", str(fc_dropout),
                                    "--head_dropout", str(head_dropout),
                                    "--patch_len", str(patch_len),
                                    "--stride", str(stride),
                                    "--des", des,
                                    "--train_epochs", str(train_epochs),
                                    "--patience", str(patience),
                                    "--lradj", lradj,
                                    "--pct_start", str(pct_start),
                                    "--itr", str(itr), 
                                    "--batch_size", str(batch_size),
                                    "--learning_rate", str(learning_rate)
                            ]

                            result = subprocess.run(command, capture_output=True, text=True)

                            print(result.stderr)

                            last_line = result.stdout.strip().split('\n')[-1].split(', ')

                            param_line = result.stdout.strip().split('\n')[1].split(', ')[1:]

                            result_line = param_line + last_line

                            with open(result_grid_path, 'a') as result_f:
                                result_f.write(f"{result_line}\n")

                            result_list.append(result_line)

                            with open(log_file_path, "w") as log_f:
                                log_f.write(result.stdout)




print(result_list)


# with open(result_grid_path, 'w') as result_f:
#     for line in result_list:
#         result_f.write(f"{line}\n")