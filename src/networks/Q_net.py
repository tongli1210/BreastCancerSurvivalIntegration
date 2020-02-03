import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_net(nn.Module):
    """
    encoder: x -> z
    """
    def __init__(self, N, x_dim, z_dim, p_drop, ngpu=1):
        super(Q_net, self).__init__()
        self.ngpu = ngpu      # number of GPU
        self.x_dim = x_dim    # dimension of input features
        self.N = N            # number of neurons in hidden layers
        self.z_dim = z_dim    # dimension of hidden variables
        self.p_drop = p_drop  # probability of dropout 

        self.main = nn.Sequential(
            nn.Linear(self.x_dim, self.N), #First layer, input -> N
            nn.Dropout(p=self.p_drop, inplace=True), #Dropout_1
            nn.ReLU(True), #ReLU_1
            nn.Linear(self.N, self.N), #Second layer, N -> N
            nn.Dropout(p=self.p_drop, inplace=True), #Dropout_2
            nn.ReLU(True), #ReLU_2
            nn.Linear(self.N, self.z_dim) #Gaussian code (z)
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            z = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            z = self.main(x)
        return z
