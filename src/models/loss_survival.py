import numpy as np
import torch
import torch.nn as nn


def get_R_matrix(survival_time):
    """
    Create an indicator matrix of risk sets, where T_j >= T_i.
    Input:
        survival_time: a Pytorch tensor that the number of rows is equal top the number of samples 
    Output:
        indicator matrix: an indicator matrix
    """
    batch_length = survival_time.shape[0]
    R_matrix = np.zeros([batch_length, batch_length], dtype=int)
    for i in range(batch_length):
        for j in range(batch_length):
            R_matrix[i, j] = survival_time[j] >= survival_time[i]
    return R_matrix

def neg_par_log_likelihood(pred, survival_time, survival_event, cuda=1):
    """
    Calculate the average Cox negative partial log-likelihood
    Input:
        pred: linear predictors from trained model.
        survival_time: survival time from ground truth
        survival_event: survival event from ground truth: 1 for event and 0 for censored
    Output:
        cost: the survival cost to be minimized
    """
    n_observed = survival_event.sum(0)
    #print(n_observed)
    R_matrix = get_R_matrix(survival_time)
    R_matrix = torch.Tensor(R_matrix)
    if cuda:
        R_matrix = R_matrix.cuda()
    risk_set_sum = R_matrix.mm(torch.exp(pred))
    #print("risk_set_sum", risk_set_sum)
    diff = pred - torch.log(risk_set_sum)
    #print("diff", diff)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(survival_event)
    #print("sum_diff_in_observed", sum_diff_in_observed)
    loss = (- (sum_diff_in_observed)/ n_observed ).reshape((-1, ))
    return loss

if __name__ == "__main__":
    survival_time = torch.Tensor([[500], [400], [1200], [300]])
    survival_event = torch.Tensor([[1], [0], [0], [1]])
    pred = torch.Tensor([[0.5], [0.3], [0.3], [0.7]])
     
    R_matrix = get_R_matrix(survival_time)
    loss = neg_par_log_likelihood(pred, survival_time, survival_event, cuda=0)
    print(loss)