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
            default='~/projects/TCGA_Omics/TCGA_BRCA_processed_data/Kfold_val_normalized_selected',
            help="the data folder with csv files" 
        )
        parser.add_argument(
            '--fold',
            default=0,
            help="the fold" 
        )
        parser.add_argument(
            '--modality',
            default='GeneExp',
            help="the data modality"
        )
        return parser, set()
      
    def __init__(self, opt, mode):
        if mode == 'train':
            self.data_csv = join(opt.data_folder, opt.modality+'_Train'+str(opt.fold)+'.csv') 
            self.label_csv = join(opt.data_folder, 'Label_Train'+str(opt.fold)+'.csv') 
        elif mode == 'val':
            self.data_csv = join(opt.data_folder, opt.modality+'_Val'+str(opt.fold)+'.csv') 
            self.label_csv = join(opt.data_folder, 'Label_Val'+str(opt.fold)+'.csv')
        elif mode == 'test':
            self.data_csv = join(opt.data_folder, opt.modality+'_Test'+str(opt.fold)+'.csv')  
            self.label_csv = join(opt.data_folder, 'Label_Test'+str(opt.fold)+'.csv')
   
        self.modality = opt.modality
        self.df_feature = prepareFeature(self.data_csv) 
        self.df_label = prepareLabel(self.label_csv)

    def __len__(self):
        assert len(self.df_feature.index) == len(self.df_label.index)
        return len(self.df_feature.index) # get the number of rows (samples)

    def __getitem__(self, idx):
        #prepare survival data
        data = {'label': torch.Tensor([self.df_label.iloc[idx]['label']])} # binary label
        data['event'] = torch.Tensor([self.df_label.iloc[idx]['event']]) # event
        if data['event'] == 0:
            data['days'] = torch.Tensor([self.df_label.iloc[idx]['days_to_last_follow_up']])
        else:
            data['days'] = torch.Tensor([self.df_label.iloc[idx]['days_to_death']])
        #prepare X
        X = self.df_feature.iloc[idx].values # select by index
        X = [float(x) for x in X]
        X = torch.Tensor(X)
        #prepare y
        data[self.modality]= X
        return data 

