import os, sys
import torch
import torch.nn as nn
import numpy as np
from time import time
from .netinterface import NetInterface
from .loss_survival import neg_par_log_likelihood
from networks.Q_net import Q_net
from networks.P_net import P_net
from networks.C_net import C_net
from util.util_os_eval import CIndex, formatTable, CoxLogRank

class Model(NetInterface):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--net_N',
            type=int,
            default=128,
            help="Number of neurons in hidden layers"
        )        
        parser.add_argument(
            '--x_dim',
            type=int,
            default=1000,
            help="dimension of input features"
        )
        parser.add_argument(
            '--z_dim',
            type=int,
            default=100,
            help="dimension of hidden variables"
        )
        parser.add_argument(
            '--n_classes',
            type=int,
            default=1,
            help="number of nodes for classification"
        )
        parser.add_argument(
            '--p_drop',
            type=float,
            default=0.2,
            help="probability of dropout"
        )
        parser.add_argument(
            '--clf_weights',
            type=float,
            nargs='+',
            default=0.7,
            help="classification weight for each class" 
        )
        return parser, set()
    
    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        self.net_N = opt.net_N
        self.x_dim = opt.x_dim
        self.z_dim = opt.z_dim
        self.n_classes = opt.n_classes
        self.p_drop = opt.p_drop
        self.modality = opt.modality
        self.clf_weights = opt.clf_weights

        # init networks
        self.net_q = Q_net(self.net_N, self.x_dim, self.z_dim, self.p_drop)
        self.net_p = P_net(self.net_N, self.x_dim, self.z_dim, self.p_drop)
        self.net_c = C_net(self.net_N, self.z_dim, self.n_classes, self.p_drop)
        self._nets = [self.net_q, self.net_p, self.net_c]

        # optimizers
        # self.optimizer and self.optimizer_params have been initialized in netinterface
        self.optimizer_q = self.optimizer(
            self.net_q.parameters(),
            lr=opt.lr,
            **self.optimizer_params
        )
        self.optimizer_p = self.optimizer(
            self.net_p.parameters(),
            lr=opt.lr,
            **self.optimizer_params
        )
        self.optimizer_c = self.optimizer(
            self.net_c.parameters(),
            lr=opt.lr,
            **self.optimizer_params
        )
        self._optimizers = [self.optimizer_q, self.optimizer_p, self.optimizer_c]

        # schedulers
        # self.scheduler and self.scheduler_params have been initialized in netinterface
        self.scheduler_q = self.scheduler(
            self.optimizer_q,
            **self.scheduler_params
        )
        self.scheduler_p = self.scheduler(
            self.optimizer_p,
            **self.scheduler_params
        )
        self.scheduler_c = self.scheduler(
            self.optimizer_c,
            **self.scheduler_params
        )
        self._schedulers = [self.scheduler_q, self.scheduler_p, self.scheduler_c]

        # general
        self.opt =opt  
        self._metrics = ['loss_mse', 'loss_survival', 'c_index'] # log the autoencoder loss and the classification loss
        if opt.log_time:
            self._metrics += ['t_recon', 't_survival']

        # init variables
        self.init_vars(add_path=True)
        
        # init weights
        self.init_weight(self.net_q)
        self.init_weight(self.net_p)
        self.init_weight(self.net_c)

    def __str__(self):
        s = "Autoencoder"
        return s
    
    def _train_on_batch(self, epoch, batch_idx, batch):
        net_q, net_p, net_c = self.net_q, self.net_p, self.net_c
        opt_q, opt_p, opt_c = self.optimizer_q, self.optimizer_p, self.optimizer_c
        net_q.train()
        net_p.train()
        net_c.train()   

        X = batch[self.modality].cuda()
        survival_time = batch['days'].cuda()
        survival_event = batch['event'].cuda()
        batch_size = X.shape[0]
        batch_log = {'size': batch_size}

        # Stage 1: train Q and P with reconstruction loss
        net_q.zero_grad()
        net_p.zero_grad()
        for p in net_q.parameters():
            p.requires_grad=True
        for p in net_p.parameters():
            p.requires_grad=True
        for p in net_c.parameters():
            p.requires_grad=False
        t0 = time()
        z = net_q(X)
        X_recon = net_p(z)
        loss_mse = nn.functional.mse_loss(X_recon, X) # Mean square error
        loss_mse.backward()
        opt_q.step()
        opt_p.step()
        t_recon = time() - t0
        batch_log['loss_mse'] = loss_mse.item() 

        # Stage 2: train Q and C with classification loss
        if not survival_event.sum(0): # skip the batch if all instances are negative
            batch_log['loss_survial'] = torch.Tensor([float('nan')])
            return batch_log
        net_q.zero_grad()
        net_c.zero_grad()
        for p in net_q.parameters():
            p.requires_grad=True
        for p in net_c.parameters():
            p.requires_grad=True
        for p in net_p.parameters():
            p.requires_grad=False
        t0 = time()
        z = net_q(X)
        pred = net_c(z)
        loss_survival = neg_par_log_likelihood(pred, survival_time, survival_event)
        loss_survival.backward()
        opt_q.step()
        opt_c.step()
        t_survival = time() - t0
        c_index = CIndex(pred, survival_event, survival_time)
        batch_log['loss_survival'] = loss_survival.item()
        batch_log['c_index'] = c_index
        if self.opt.log_time:
            batch_log['t_recon'] = t_recon
            batch_log['t_survival'] = t_survival
        return batch_log

    def _vali_on_batch(self, epoch, batch_idx, batch):
        self.net_q.eval()
        self.net_c.eval()
        # load the data
        X = batch[self.modality].cuda()
        survival_time = batch['days'].cuda()
        survival_event = batch['event'].cuda()
        batch_size = X.shape[0]
        batch_log = {'size': batch_size}
        if not survival_event.sum(0): # skip the batch if all instances are negative
            batch_log['loss_survival'] = torch.Tensor([float('nan')])
            return batch_log
        with torch.no_grad():
            z = self.net_q(X)
            pred = self.net_c(z)
        loss_survival = neg_par_log_likelihood(pred, survival_time, survival_event)
        c_index = CIndex(pred, survival_event, survival_time)
        batch_log['loss'] = loss_survival.item()
        batch_log['loss_survival'] = loss_survival.item()
        batch_log['c_index'] = c_index
        return batch_log


class Model_test(Model):
    @classmethod
    def add_arguments(cls, parser):
        parser, unique_params = Model.add_arguments(parser)
        return parser, unique_params

    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        self.load_state_dict(opt.net_file, load_optimizer='auto')
        self.output_dir = opt.output_dir 

    def __str__(self):
        return "Testing Single Modality Autoencoder"    
        
    def test_on_batch(self, batch_ind, batch):
        outdir = os.path.join(self.output_dir, 'batch%04d' % batch_ind)
        os.makedirs(outdir, exist_ok=True) 
        self.net_q.eval()
        self.net_c.eval()
        X = batch[self.modality].cuda()
        survival_time = batch['days'].cuda()
        survival_event = batch['event'].cuda()
        #batch_size = X.shape[0]
        #batch_log = {'size': batch_size}
 
        with torch.no_grad():
            z = self.net_q(X)
            pred = self.net_c(z)
        
        loss = neg_par_log_likelihood(pred, survival_time, survival_event)
        print('loss', loss.item())
        c_index = CIndex(pred, survival_event, survival_time)
        coxLogRank_p = CoxLogRank(pred, survival_event, survival_time)
        eva = {'c_index': c_index, 'coxLogRank_p': coxLogRank_p}
        for key in eva:
            print(key, eva[key])
        formatTable(eva, outpath=os.path.join(outdir, 'eva.csv'))        
        output = self.pack_output(pred, batch)
        np.savez(os.path.join(outdir, 'batch%04d.npz' % batch_ind), **output)

    def pack_output(self, pred, batch):
        out = {}
        out['pred'] = pred.detach().cpu().numpy()
        out['event'] = batch['event'].cpu().numpy()
        out['days'] = batch['days'].cpu().numpy()
        return out
