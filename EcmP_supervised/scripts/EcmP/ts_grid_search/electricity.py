import os 
import subprocess
from pathlib import Path

import pandas as pd

#Directory 
result_dir = "./logs/" + "EcmP_gridsearch"
log_dirs = ["./logs", result_dir]
for dir_path in log_dirs:
    os.makedirs(dir_path, exist_ok=True)

#logistics
model_name = "EcmP"
result_log_path = "./result_log/EcmP/electricity.txt"



#dataset components
root_path_name = "./data/ts_benchmark"
data_path_name = "electricity.csv"
model_id_name = "Electricity"
data_name = "custom"

features = "M"
enc_in = 321


#model components
dropout = 0.2
fc_dropout = 0.2
head_dropout = 0


#training components
des = 'Exp' 
train_epochs = 100
patience = 10
lradj = 'TST'
pct_start = 0.2
itr = 1
batch_size = 32
learning_rate = 0.0001

#patching setting
# patch_len = 16
# stride = 8


#Parameter grid
seq_lens = [336]
pred_lens = [96, 192, 336, 720]

e_layers = [2, 3, 4, 5]
n_heads = [4, 8]


model_size_multiplier = 3

model_var_preset = [

    {
        "d_patch" : 16,
        "d_model" : 128,
        "d_ff" : 256
    }

    ,

    {
        "d_patch" : 32,
        "d_model" : 256,
        "d_ff" : 256
    }

]

patching_size_multiplier = 3

patching_var_preset = [

    {
        "patch_len" : 8,
        "stride" : 4
    }

]



#



