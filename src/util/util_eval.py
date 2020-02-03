import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import pandas as pd

def reportMetrics(target, pred_prob, threshold=0.5):
    """
    target: tensor
    pred_prob: pred in probability
    """
    eva = dict()
    target = target.cpu()
    pred_prob = pred_prob.cpu()
    # binarized the prediction 
    pred = torch.where(pred_prob >= threshold, 
                       torch.ones(pred_prob.size()), 
                       torch.zeros(pred_prob.size()))    

    # move the data to numpy
    target_np = target.numpy()
    pred_prob_np = pred_prob.numpy()
    pred_np = pred.numpy()
    
    eva['acc'] = accuracy_score(target_np, pred_np)
    eva['precision'] = precision_score(target_np, pred_np)
    eva['recall'] = recall_score(target_np, pred_np)

    fpr, tpr, _ = roc_curve(target_np, pred_prob_np)
    eva['auc'] = auc(fpr, tpr)
    #print('Accuracy: {}'.format(eva['acc']))
    print(target_np, pred_prob_np, pred_np)
    print(eva)
    return eva

def reportMetricsMultiClass(target, pred_prob, threshold=0.5):
    """
    target: tensor
    pred_prob: pred in probability
    """
    eva = dict()
    target = target.cpu()
    pred_prob = pred_prob.cpu()  # shape (n_samples , n_classes)
    # binarized the prediction 
    pred = np.argmax(pred_prob, axis=-1) 

    # move the data to numpy
    target_np = target.numpy()  # shape (n_samples, 1) with elements in 0, 1, .., n_classes-1
    pred_prob_np = pred_prob.numpy()
    pred_np = pred.numpy()  # shape (n_samples, 1) with elements in 0, 1, ..., n_classes-1 
    
    print(target_np.shape)
    print(pred_prob_np.shape)
    print(pred.shape)
    eva['acc'] = accuracy_score(target_np, pred_np)
    eva['precision'] = precision_score(target_np, pred_np, average='weighted')
    eva['recall'] = recall_score(target_np, pred_np, average='weighted')
    # No ROC curve for multi-class classification

    print(target_np, pred_prob_np, pred_np)
    print(eva)
    return eva

def formatTable(eva, outpath):
    df = pd.DataFrame(eva, index=[0])
    df.to_csv(outpath, sep=',', header=True, index=False)
    return df

def formatCombinedTable(eva, outFile='metrics.csv'):
    df = pd.DataFrame.from_dict(eva)
    df_stat=df.append(df.mean().rename('Mean'))
    df_stat=df_stat.append(df.std().rename('Std'))
    formatedOutput = [x+'±'+y for x,y in zip(df_stat.loc['Mean'].round(3).astype(str), df_stat.loc['Std'].round(3).astype(str))]
    formatedOutput = pd.Series(formatedOutput,index=df_stat.columns.values)
    formatedOutput = formatedOutput.rename('MeanStd')
    df_stat = df_stat.append(formatedOutput)
    df_stat.to_csv(outputFile, sep=',', header=True, index=True)
    return df_stat
