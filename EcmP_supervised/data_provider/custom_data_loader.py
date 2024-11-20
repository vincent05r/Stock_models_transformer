import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import pickle


class Dataset_Custom_stock(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='stock_000001.SZ.csv',
                 target='close_pct_chg', scale=0, timeenc=1, freq='d', dt_format_str=0): 
        
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.date_str = 'date' #'trade_date'

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        #customize dt format string
        dt_f_str_dict = {1 : '%Y%m%d'}
        if dt_format_str != 0:
            dt_format_str = dt_f_str_dict[dt_format_str]
        self.dt_format_str = dt_format_str #the formatting string for pandas datetime function

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['trade_date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(self.date_str)
        df_raw = df_raw[[self.date_str] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len] #[train_begin, val_begin, test_begin]
        border2s = [num_train, num_train + num_vali, len(df_raw)]  #[train_end, val_end, test_end]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale: #if scalling
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values) #use std and mean from training set.
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.date_str]][border1:border2]

        if self.dt_format_str == 0: #be aware all date column must be named date
            df_stamp[self.date_str] = pd.to_datetime(df_stamp.date) #pandas will convert the column name directly into a self.column name for accessing (similar)
        else:
            df_stamp[self.date_str] = pd.to_datetime(df_stamp.date, format=self.dt_format_str)

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # data_stamp = df_stamp.drop([self.date_str], axis=1).values
            print("nah bro it doesnt work, it is only for transformer")
            raise(Exception)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.date_str].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        if self.scale:
            # scaler result save
            folder_path = './scaler/' + self.data_path.replace('.csv', '') + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            with open( folder_path + self.data_path.replace('.csv', '') + '_' + str(self.seq_len) + '_' + str(self.pred_len) + '.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

            #np.save( folder_path + self.data_path.replace('.csv', '') + '_' + str(self.seq_len) + '_' + str(self.pred_len) , np.array([self.scaler.mean_, self.scaler.var_]) )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


#todo
class Dataset_Custom_stock_pred(Dataset):


    def __init__(self, root_path, flag='pred', size=None,
                features='MS', data_path='stock_000001.SZ.csv',
                target='close_pct_chg', scale=0, prev_scaler='None', inverse=False, timeenc=1, freq='d', dt_format_str=0, cols=None): 


        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']


        self.date_str = 'date' #'trade_date'

        self.features = features
        self.target = target
        self.scale = scale
        self.prev_scaler = prev_scaler
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = 0
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:

            #swap to exisitng scaler
            if self.prev_scaler != 'None':
                with open(self.prev_scaler, 'rb') as f:
                    self.scaler = pickle.load(f)

            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
                 
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_stock_pretrain(Dataset):
    def __init__(self, root_path, flag='pre_train', size=None,
                 features='MS', data_path='stock_000001.SZ.csv',
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

        self.date_str = 'date' #'trade_date'

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        #customize dt format string
        dt_f_str_dict = {1 : '%Y%m%d'}
        if dt_format_str != 0:
            dt_format_str = dt_f_str_dict[dt_format_str]
        self.dt_format_str = dt_format_str #the formatting string for pandas datetime function

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        '''
        df_raw.columns: ['trade_date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(self.date_str)
        df_raw = df_raw[[self.date_str] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw))
        num_test = int(0)
        num_vali = int(0)
        border1s = [0] #[train_begin]
        border2s = [num_train]  #[train_end]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale: #if scalling
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values) #use std and mean from training set.
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.date_str]][border1:border2]

        if self.dt_format_str == 0: #be aware all date column must be named date
            df_stamp[self.date_str] = pd.to_datetime(df_stamp.date) #pandas will convert the column name directly into a self.column name for accessing (similar)
        else:
            df_stamp[self.date_str] = pd.to_datetime(df_stamp.date, format=self.dt_format_str)

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)  2019/05/05  05/05/2019
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # data_stamp = df_stamp.drop([self.date_str], axis=1).values
            print("nah bro it doesnt work, it is only for transformer")
            raise(Exception)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.date_str].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # if self.scale:
        #     # scaler result save
        #     folder_path = './scaler/' + self.data_path.replace('.csv', '') + '/'
        #     if not os.path.exists(folder_path):
        #         os.makedirs(folder_path)

        #     with open( folder_path + self.data_path.replace('.csv', '') + '_' + str(self.seq_len) + '_' + str(self.pred_len) + '.pkl', 'wb') as f:
        #         pickle.dump(self.scaler, f)

            #np.save( folder_path + self.data_path.replace('.csv', '') + '_' + str(self.seq_len) + '_' + str(self.pred_len) , np.array([self.scaler.mean_, self.scaler.var_]) )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)