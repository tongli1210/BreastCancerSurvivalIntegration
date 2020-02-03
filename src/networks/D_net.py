import torch
import torch.nn as nn
import torch.nn.functional as F

class D_net(nn.Module):
    """
    discriminator network
    """
    def __init__(self, para):
        super(D_net, self).__init__()
        self.ngpu = para.ngpu
        self.x_dim = para.x_dim
        self.N = para.N
        self.z_dim = para.z_dim
        self.p_drop = para.p_drop

        self.main = nn.Sequential(
            nn.Linear(self.z_dim, self.N),
            nn.Dropout(p=self.p_drop, inplace=True),
            nn.ReLU(True),
            nn.Linear(self.N, self.N),
            nn.Dropout(p=self.p_drop, inplace=True),
            nn.ReLU(True),
            nn.Linear(self.N, 2),
            nn.Sigmoid()
        )

    def forward(self, z):
        if isinstance(z.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            decision = nn.parallel.data_parallel(self.main, z, range(self.ngpu))
        else:
            decision = self.main(z)
        return decision
