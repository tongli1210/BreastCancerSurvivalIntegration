import os, sys
import torch
import torch.nn as nn
import numpy as np
from time import time
from .netinterface import NetInterface
from networks.Q_net import Q_net
from networks.C_net import C_net
from util.util_eval import reportMetricsMultiClass, formatTable

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
        self.modality = opt.modality[0] # get the fisrt element since changed this argument to list
        self.clf_weights = opt.clf_weights

        # init networks
        self.net_q = Q_net(self.net_N, self.x_dim, self.z_dim, self.p_drop)
        self.net_c = C_net(self.net_N, self.z_dim, self.n_classes, self.p_drop)
        self._nets = [self.net_q, self.net_c]

        # optimizers
        # self.optimizer and self.optimizer_params have been initialized in netinterface
        self.optimizer_q = self.optimizer(
            self.net_q.parameters(),
            lr=opt.lr,
            **self.optimizer_params
        )
        self.optimizer_c = self.optimizer(
            self.net_c.parameters(),
            lr=opt.lr,
            **self.optimizer_params
        )
        self._optimizers = [self.optimizer_q, self.optimizer_c]

        # schedulers
        # self.scheduler and self.scheduler_params have been initialized in netinterface
        self.scheduler_q = self.scheduler(
            self.optimizer_q,
            **self.scheduler_params
        )
        self.scheduler_c = self.scheduler(
            self.optimizer_c,
            **self.scheduler_params
        )
        self._schedulers = [self.scheduler_q, self.scheduler_c]

        # general
        self.opt =opt  
        self._metrics = ['loss_clf'] # log the autoencoder loss and the classification loss
        if opt.log_time:
            self._metrics += ['t_clf']

        # init variables
        self.init_vars(add_path=True)
        
        # init weights
        self.init_weight(self.net_q)
        self.init_weight(self.net_c)

    def __str__(self):
        s = "Autoencoder"
        return s
    
    def _train_on_batch(self, epoch, batch_idx, batch):
        net_q, net_c = self.net_q, self.net_c
        opt_q, opt_c = self.optimizer_q, self.optimizer_c
        net_q.train()
        net_c.train()   

        X = batch[self.modality].cuda()
        #print(X.shape) # (batchsize, 1, 28, 28)
        # Flatten the X from 28x28 to 784
        X = X.view(X.shape[0], -1) 
        #print(X.shape) # (batchsize, 784)
        y = batch['labels'].cuda()
        batch_size = X.shape[0]
        batch_log = {'size': batch_size}
        

        net_q.zero_grad()
        net_c.zero_grad()
        for p in net_q.parameters():
            p.requires_grad=True
        for p in net_c.parameters():
            p.requires_grad=True
        t0 = time()
        z = net_q(X)
        pred = net_c(z)

        criterion = nn.CrossEntropyLoss() 
        # the cross-entropy loss combines the 1)LogSoftmax() and 2) Negative log-likelihood loss NLLLoss
        loss_clf = criterion(pred, y) 
        loss_clf.backward()
        opt_q.step()
        opt_c.step()
        t_clf = time() - t0
        batch_log['loss_clf'] = loss_clf.item()
        if self.opt.log_time:
            batch_log['t_clf'] = t_clf
        return batch_log

    def _vali_on_batch(self, epoch, batch_idx, batch):
        self.net_q.eval()
        self.net_c.eval()
        X = batch[self.modality].cuda()
        X = X.view(X.shape[0], -1)  # flatten the X
        y = batch['labels'].cuda()
        batch_size = X.shape[0]
        batch_log = {'size': batch_size}
        with torch.no_grad():
            z = self.net_q(X)
            pred = self.net_c(z)

        criterion = nn.CrossEntropyLoss() 
        # the cross-entropy loss combines the 1)LogSoftmax() and 2) Negative log-likelihood loss NLLLoss
        loss_clf = criterion(pred, y) 
        batch_log['loss'] = loss_clf.item()
        batch_log['loss_clf'] = loss_clf.item()
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
        logSoftmax = nn.LogSoftmax()
        outdir = os.path.join(self.output_dir, 'batch%04d' % batch_ind)
        os.makedirs(outdir, exist_ok=True) 
        self.net_q.eval()
        self.net_c.eval()
        X = batch[self.modality].cuda()
        X = X.view(X.shape[0], -1)
        y = batch['labels'].cuda()
        with torch.no_grad():
            z = self.net_q(X)
            pred_logits = self.net_c(z)
            pred_prob = logSoftmax(pred_logits) 
        eva = reportMetricsMultiClass(y, pred_prob)
        formatTable(eva, outpath=os.path.join(outdir, 'eva.csv'))
        output = self.pack_output(pred_prob, batch)
        np.savez(os.path.join(outdir, 'batch%04d.npz' % batch_ind), **output)

    def pack_output(self, pred_prob, batch):
        out = {}
        out['pred_prob'] = pred_prob.detach().cpu().numpy()
        out['target'] = batch['labels'].numpy()
        return out    
