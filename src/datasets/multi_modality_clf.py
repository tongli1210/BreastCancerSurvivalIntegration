from os.path import join
import random
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch
from .dataset_util import *

class Dataset(data.Dataset):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--data_folder',
            default='/home/ltong/projects/TCGA_Omics/TCGA_CancerType/Processed/Kfold_val_normalized_selected',
            help="the data folder with csv files" 
        )
        parser.add_argument(
            '--fold',
            default=0,
            help="the fold" 
        )
        parser.add_argument(
            '--modality',
            nargs='+',
            default='GeneExp',
            help="the data modality, takes one or more arguments for integration"
        )
        return parser, set()
      
    def __init__(self, opt, mode):
        self.modality = opt.modality
        self.n_modality = len(self.modality)  
        self.data_csv = [None]*self.n_modality
        self.df_feature = [None]*self.n_modality
        if mode == 'train':
            for i in range(self.n_modality):
                self.data_csv[i] = join(opt.data_folder, opt.modality[i]+'_Train'+str(opt.fold)+'.csv') 
            self.label_csv = join(opt.data_folder, 'Label_Train'+str(opt.fold)+'.csv') 
        elif mode == 'val':
            for i in range(self.n_modality):
                self.data_csv[i] = join(opt.data_folder, opt.modality[i]+'_Val'+str(opt.fold)+'.csv') 
            self.label_csv = join(opt.data_folder, 'Label_Val'+str(opt.fold)+'.csv')
        elif mode == 'test':
            for i in range(self.n_modality):
                self.data_csv[i] = join(opt.data_folder, opt.modality[i]+'_Test'+str(opt.fold)+'.csv')  
            self.label_csv = join(opt.data_folder, 'Label_Test'+str(opt.fold)+'.csv')
        for i in range(self.n_modality):
            self.df_feature[i] = prepareFeature(self.data_csv[i]) 
        self.df_label = pd.read_csv(self.label_csv, header=0, index_col=None)

    def __len__(self):
        for i in range(self.n_modality):
            assert len(self.df_feature[i].index) == len(self.df_label.index)
        return len(self.df_label.index) # get the number of rows (samples)

    def __getitem__(self, idx):
        #prepare survival data
        data = {'labels': torch.Tensor([self.df_label.iloc[idx]['label']])} # binary label
        #prepare X
        for i in range(self.n_modality):
            X = self.df_feature[i].iloc[idx].values # select by index
            X = [float(x) for x in X]
            X = torch.Tensor(X)
            data[self.modality[i]] = X
        return data

