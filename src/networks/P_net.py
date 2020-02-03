import torch
import torch.nn as nn
import torch.nn.functional as F

class P_net(nn.Module):
    """
    Decoder: z -> x
    """
    def __init__(self, N, x_dim, z_dim, p_drop, ngpu=1):
        super(P_net, self).__init__()
        self.ngpu = ngpu      # number of GPU
        self.x_dim = x_dim    # dimension of input features
        self.N = N            # number of neurons in hidden layers 
        self.z_dim = z_dim    # dimension of hidden variables
        self.p_drop = p_drop  # probability of dropout

        self.main = nn.Sequential(
            nn.Linear(self.z_dim, self.N),
            nn.Dropout(p=self.p_drop, inplace=True), #Dropout_1
            nn.ReLU(True), #ReLU_1
            nn.Linear(self.N, self.N),
            nn.Dropout(p=self.p_drop, inplace=True), #Dropout_2
            #nn.ReLU(True),
            nn.Linear(self.N, self.x_dim),
        )

    def forward(self, z):
        if isinstance(z.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            x_recon = nn.parallel.data_parallel(self.main, z, range(self.ngpu))
        else:
            x_recon = self.main(z)
        return x_recon
