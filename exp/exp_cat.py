from data.data_loader import Dataset_HDFS
from exp.exp_basic import Exp_Basic
from models.model import CAT
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_CAT(Exp_Basic):
    def __init__(self, args):
        super(Exp_CAT, self).__init__(args)
        self.center = torch.zeros(self.args.c_out).to(self.device)
        self.radius = torch.tensor(0).to(self.device)
    
    def _build_model(self):
        model = CAT(
            self.args.enc_in,
            self.args.dec_in, 
            self.args.c_out, 
            self.args.seq_len, 
            self.args.label_len,
            self.args.pred_len, 
            self.args.factor,
            self.args.d_model, 
            self.args.n_heads, 
            self.args.e_layers,
            self.args.d_layers, 
            self.args.d_ff,
            self.args.dropout, 
            self.args.attn,
            self.args.embed,
            # self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'HDFS':Dataset_HDFS
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len]
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, flag = None):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y) in enumerate(vali_loader):
            pred, true,oc = self._process_one_batch(
                vali_data, batch_x, batch_y)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            oc_loss = torch.mean(torch.sum((oc - self.center) ** 2, dim = 1)).detach().cpu()
            loss += oc_loss
            total_loss.append(loss)
        total_loss = np.average(total_loss)

        return total_loss

    def update_center(self, train_loader):
        self.model.eval()
        result = torch.zeros(self.args.c_out, device=self.device)
        n_samples = 0
        radius = torch.tensor(0)
        with torch.no_grad():
            for i,(batch_x, batch_y) in enumerate(train_loader):
                _, _, oc = self._process_one_batch(train_loader, batch_x, batch_y)
                n_samples += oc.size(0)
                result += torch.sum(oc, dim = 0)
                cur_radius = torch.max(torch.sum((oc-self.center)**2, dim = 1))
                if (cur_radius > radius):
                    radius = cur_radius
        result /= n_samples
        return result, radius


    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_normal_data, test_normal_loader = self._get_data(flag = 'test_normal')
        test_anomaly_data, test_anomaly_loader = self._get_data(flag = 'test_anomaly')

        self.center, self.radius = self.update_center(train_loader)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true, oc = self._process_one_batch(
                    train_data, batch_x, batch_y)
                mse_loss = criterion(pred, true)
                oc_loss = torch.sum((oc - self.center) ** 2, dim = 1)
                mean_loss = torch.mean(oc_loss)
                loss = mean_loss + mse_loss
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
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

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_normal_loss = self.vali(test_normal_data, test_normal_loader, criterion, flag = 'test_normal')
            test_anomaly_loss = self.vali(test_anomaly_data, test_anomaly_loader, criterion, flag = 'test_abnormal')

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test normal Loss: {4:.7f} Test anomaly Loss: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_normal_loss, test_anomaly_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))


        
        return self.model

    def test(self, setting):
        test_normal_data, test_normal_loader = self._get_data(flag = 'test_normal')
        test_anomaly_data, test_anomaly_loader = self._get_data(flag = 'test_anomaly')
        valid_data, valid_loader = self._get_data(flag = 'val')
        
        self.model.eval()
        
        preds = []
        trues = []
        ocs_val = []
        ocs_normal = []
        ocs_anomaly = []
        for i, (batch_x,batch_y) in enumerate(valid_loader):
            pred, true, oc = self._process_one_batch(
                valid_data, batch_x, batch_y)
            ocs_val.append((torch.sum((oc - self.center)**2, dim = 1) - self.radius).detach().cpu().numpy())
        ocs_val = np.array(ocs_val).reshape(-1)
        threshold = np.percentile(ocs_val, 90)
        
        for i, (batch_x,batch_y) in enumerate(test_normal_loader):
            pred, true, oc = self._process_one_batch(
                test_normal_data, batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            ocs_normal.append((torch.sum((oc - self.center)**2, dim = 1) - self.radius).detach().cpu().numpy())

        for i, (batch_x,batch_y) in enumerate(test_anomaly_loader):
            pred, true, oc = self._process_one_batch(
                test_anomaly_data, batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            ocs_anomaly.append((torch.sum((oc - self.center)**2, dim = 1) - self.radius).detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        ocs_normal = np.array(ocs_normal)
        ocs_anomaly = np.array(ocs_anomaly)

        print('test shape:', preds.shape, trues.shape, ocs_normal.shape, ocs_anomaly.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        ocs_normal = ocs_normal.reshape(-1)
        ocs_anomaly = ocs_anomaly.reshape(-1)
        print('test shape:', preds.shape, trues.shape, ocs_normal.shape, ocs_anomaly.shape)

        TN, FP = np.sum(np.where(ocs_normal <= threshold, 1, 0)), np.sum(np.where(ocs_normal > threshold, 1, 0))
        TP, FN = np.sum(np.where(ocs_anomaly > threshold, 1, 0)), np.sum(np.where(ocs_anomaly <= threshold, 1, 0))

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        from sklearn.metrics import roc_auc_score, average_precision_score
        y_true = []
        y_pred = []
        for i in range(ocs_normal.shape[0]):
            y_true.append(0)
            # print(ocs_normal[i])
            y_pred.append(ocs_normal[i])
        for i in range(ocs_anomaly.shape[0]):
            y_true.append(1)
            y_pred.append(ocs_anomaly[i])
        print(TN, TP, FP, FN, P, R, F1, roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred))

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true, oc = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        #add one for [SEQ] token
        dec_inp = torch.cat([batch_y[:,:self.args.label_len+1,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs, attention, true = self.model(batch_x, dec_inp)
                else:
                    outputs = self.model(batch_x, dec_inp)
        else:
            if self.args.output_attention:
                outputs, attention = self.model(batch_x, dec_inp)
            else:
                outputs = self.model(batch_x, dec_inp)
        true = dec_inp[:,-self.args.pred_len:,:].to(self.device)
        oc = outputs[:,0,:].to(self.device)
        return outputs, true, oc

