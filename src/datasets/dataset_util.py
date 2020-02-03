import pandas as pd

def prepareFeature(DataFile):
    df_feature = pd.read_csv(DataFile, sep=',', header=0, index_col=0)
    df_feature = df_feature.dropna(axis=0, how='any')
    df_feature = df_feature.T
    return df_feature

def prepareLabel(LabelFile):
    df_label = pd.read_csv(LabelFile, sep=',', header=0, index_col=None)
    y = df_label['vital_status'].map({'alive':0, 'dead': 1})
    df_label['event'] = y  # 1 for event/dead, 0 for no event/alive
    return df_label