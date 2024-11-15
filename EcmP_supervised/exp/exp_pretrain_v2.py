from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, EcmP, EcmP_mk2, EcmP_mk3
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import wandb

warnings.filterwarnings('ignore')

class Exp_Pretrain_v2(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model_dict = {
            'EcmP' : EcmP,
            'EcmP_mk2' : EcmP_mk2,
            'EcmP_mk3' : EcmP_mk3
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='pre_train')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = path + '/' + 'checkpoint.pth'

        if self.args.model_load_path == 'None':
            #get initialise a new pretrain model
            pass
        else:
            #load the model and keep training
            self.model.load_state_dict(torch.load(model_path))  #todo, set a loading path for the pretrainig model

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        

        #wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="PCIE-pretrain",
            # track hyperparameters and run metadata
            config={
            "learning_rate": self.args.learning_rate,
            "architecture": self.args.model,
            "dataset": self.args.root_path,
            "epochs": self.args.train_epochs,
            "d_model": self.args.d_model,
            "d_ff": self.args.d_ff,
            "enc_in": self.args.enc_in,
            "e_layers": self.args.e_layers,
            "n_heads": self.args.n_heads,
            "patch_len": self.args.patch_len,
            "stride": self.args.stride,
            "batch_size": self.args.batch_size,
            "first_stage_patching": self.args.first_stage_patching,
            "second_stage_patching": self.args.second_stage_patching,
            "seq_len": self.args.seq_len,
            "pred_len": self.args.pred_len,
            "label_len": self.args.label_len,
            "scale": self.args.scale,
            "random_seed": self.args.random_seed,
            }
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():

                        outputs = self.model(batch_x)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)  #important part, MS will only calculate the loss based on the predicted series
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


            torch.save(self.model.state_dict(), model_path)
            print("Saving pretrain model on Epoch {}".format(epoch))

            wandb.log({"loss": train_loss})


        wandb.finish()

        return train_loss
