import torch
import torch.nn as nn
import torch.nn.functional as F

class C_net(nn.Module):
    """
    classificaiton network with logits (no sigmoid)
    """
    def __init__(self, N, z_dim, n_classes, p_drop, ngpu=1):
        super(C_net, self).__init__()
        self.ngpu = ngpu
        self.N = N             # number of neurons in hidden layers
        self.z_dim = z_dim     # dimension of hidden variables
        self.p_drop = p_drop   # probability of dropout
        self.n_classes = n_classes  # number of classes

        self.main = nn.Sequential(
            nn.Linear(self.z_dim, self.N),
            nn.Dropout(p=self.p_drop, inplace=True),
            nn.ReLU(True),
            nn.Linear(self.N, self.N),
            nn.Dropout(p=self.p_drop, inplace=True),
            nn.ReLU(True),
            nn.Linear(self.N, self.n_classes),
        )

    def forward(self, z):
        if isinstance(z.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            decision = nn.parallel.data_parallel(self.main, z, range(self.ngpu))
        else:
            decision = self.main(z)
        return decision
