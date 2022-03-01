import os
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_HDFS(Dataset):
    def __init__(self, root_path, flag='train', size=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 48
            self.label_len = 24
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test_normal', 'val', 'test_anomaly']
        type_map = {'train':0, 'val':1, 'test_normal':2, 'test_anomaly':3}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        if (self.set_type == 0):
            df_raw = pd.read_csv(os.path.join(self.root_path, 'train_sequence.csv'))    
        if (self.set_type == 1):
            df_raw = pd.read_csv(os.path.join(self.root_path, 'valid_sequence.csv'))    
        if (self.set_type == 2):
            df_raw = pd.read_csv(os.path.join(self.root_path, 'test_normal_sequence.csv'))    
        if (self.set_type == 3):
            df_raw = pd.read_csv(os.path.join(self.root_path, 'test_anomaly_sequence.csv'))    

        # template_embeddings = torch.load(os.path.join(self.root_path, 'template_embeddings.pt'))
        # import json
        # template_hashs = json.load(open(os.path.join(self.root_path, 'hdfs_log_templates.json'),'r'))
        # template_hashs = {str(template_hashs[x]):x for i,x in enumerate(template_hashs)}
        # hash_idx = json.load(open(os.path.join(self.root_path, 'hashes.txt'), 'r'))
        data = df_raw['EventSequence'].str[2:-2].str.split("', '").values
        # data = df_raw['EventSequence'].str[2:-2].str.split("', '")
        # data = [torch.tensor(x, dtype = torch.long) for x in data]
        # print(data[0].size())

        self.data_x = data
        self.data_y = data
        # print(len(data), len(data[100]))
        # self.template_embeddings = template_embeddings
        # self.template_hashs = template_hashs
        # print(self.template_embeddings.keys())
    
    def __getitem__(self, index):

        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_token = torch.tensor(torch.rand(1,1))
        d = {str(x):x for x in range(1000)}
        d['[PAD]'] = -1
        seq_x = torch.stack([torch.tensor(d[x]).unsqueeze(0) for x in self.data_x[index][s_begin:s_end]])
        seq_y = torch.stack([torch.tensor(d[x]).unsqueeze(0) for x in self.data_y[index][r_begin:r_end]])


        # seq_token = torch.tensor(self.template_embeddings['[SEQ]'], dtype = torch.float).unsqueeze(0)

        # seq_x = torch.stack([self.template_embeddings[x] for x in self.data_x[index][s_begin:s_end]])
        # seq_y = torch.stack([self.template_embeddings[x] for x in self.data_y[index][r_begin:r_end]])

        seq_y = torch.cat([seq_token, seq_y], dim = 0)
        # print(index, s_begin, s_end)

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x)
