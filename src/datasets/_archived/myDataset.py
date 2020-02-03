import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class myDataset(Dataset):
    def __init__(self, df_feature, df_label):
        self.df_feature = df_feature
        self.df_label = df_label

    def __len__(self):
        assert len(self.df_feature.index) == len(self.df_label.index)
        return len(self.df_feature.index) # get the number of rows (samples)
    
    def __getitem__(self, idx):
        #prepare X
        X = self.df_feature.iloc[idx].values # select by index
        X = [float(x) for x in X]
        X = torch.FloatTensor(X) 
        #prepare y
        y = self.df_label['label'].values[idx]
        
        return X,y
     
if __name__ == "__main__":
    #prepare data
    DnaMethFile = "/home/ltong/projects/ADNI/TCGA_OV_processed_data/DnaMeth/DnaMeth_Selected.csv"
    GeneExpFile = "/home/ltong/projects/ADNI/TCGA_OV_processed_data/GeneExp/GeneExp_Selected.csv"
    LabelFile = "/home/ltong/projects/ADNI/TCGA_OV_processed_data/label/Label_Selected.csv"

    df_DnaMeth = pd.read_csv(DnaMethFile, sep=',', header=0, index_col=0)  # Features x Samples
    df_GeneExp = pd.read_csv(GeneExpFile, sep=',', header=0, index_col=0)  # Features x Samples
    df_Label = pd.read_csv(LabelFile, sep=',', header=0, index_col=None)  # Samples x Features

    # Drop NaN
    df_DnaMeth = df_DnaMeth.dropna(axis=0, how='any')
    df_GeneExp = df_GeneExp.dropna(axis=0, how='any')

    # Concatenate
    df_concat = pd.concat([df_GeneExp, df_DnaMeth], axis=0)
    df_concat = df_concat.T
    df_concat = df_DnaMeth.T 
    
    #prepare dataset loader
    my_dataset = myDataset(df_concat, df_Label)
    count = 0
    for X, y in my_dataset:
        #print(X.shape)
        #print(y.shape)
        count += 1
        break
    print(count)
       
    dataset_loader = torch.utils.data.DataLoader(my_dataset,
                                                 batch_size=10,
                                                 shuffle=True)
    count = 0
    for X, y in dataset_loader:
        print(X.shape)
        print(y.shape)
        count += 1
    print(count)
