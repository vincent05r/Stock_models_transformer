import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import pickle

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import glob


class Dataset_Custom_stock_pretrain_v2(Dataset):
    def __init__(self, root_path, flag='pre_train', size=None,
                 features='MS', data_folder=None,
                 target='close_pct_chg', scale=0, timeenc=1, freq='d', dt_format_str=0):

        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['pre_train']
        type_map = {'pre_train': 0}
        self.set_type = type_map[flag]

        self.date_str = 'date'  # or 'trade_date'

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # Customize date format string
        dt_f_str_dict = {1: '%Y%m%d'}
        if dt_format_str != 0:
            dt_format_str = dt_f_str_dict[dt_format_str]
        self.dt_format_str = dt_format_str  # The formatting string for pandas datetime function

        self.data_folder = data_folder  # Changed from data_path to data_folder
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Get all CSV files in the data_folder
        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        
        self.data_x = []
        self.data_y = []
        self.data_stamp = []

        for filename in all_files:
            df_raw = pd.read_csv(filename)

            # Ensure that the target and date columns are present
            cols = list(df_raw.columns)
            if self.target not in cols or self.date_str not in cols:
                raise ValueError(f"Columns '{self.target}' or '{self.date_str}' not found in the data.")
            cols.remove(self.target)
            cols.remove(self.date_str)
            df_raw = df_raw[[self.date_str] + cols + [self.target]]

            # Convert date column to datetime and sort
            if self.dt_format_str == 0:
                df_raw[self.date_str] = pd.to_datetime(df_raw[self.date_str])
            else:
                df_raw[self.date_str] = pd.to_datetime(df_raw[self.date_str], format=self.dt_format_str)
            df_raw = df_raw.sort_values(by=self.date_str).reset_index(drop=True)
            
            # Set the date column as the index
            df_raw.set_index(self.date_str, inplace=True)

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            # Scale data if required
            if self.scale:
                self.scaler.fit(df_data.values)  # Scale per file
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            # Process time features
            df_stamp = df_raw[[self.date_str]]
            data_stamp = time_features(df_stamp[self.date_str], freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

            # Generate sequences within this file
            seq_length = len(df_raw)
            for i in range(seq_length - self.seq_len - self.pred_len + 1):
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = data[s_begin:s_end]
                seq_y = data[r_begin:r_end]
                seq_x_mark = data_stamp[:, s_begin:s_end].transpose(1, 0)
                seq_y_mark = data_stamp[:, r_begin:r_end].transpose(1, 0)

                self.data_x.append(seq_x)
                self.data_y.append(seq_y)
                self.data_stamp.append((seq_x_mark, seq_y_mark))

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_mark, seq_y_mark = self.data_stamp[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
