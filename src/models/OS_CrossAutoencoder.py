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
from util.util_os_eval import CIndex, formatTable

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
            nargs='+',
            default=1000,
            help="dimension of input features, take one or more arguments for integration"
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
        return parser, set()
    
    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        self.net_N = opt.net_N
        self.x_dim = opt.x_dim
        self.z_dim = opt.z_dim
        self.n_classes = opt.n_classes
        self.p_drop = opt.p_drop
        self.modality = opt.modality
        self.n_modality = len(self.modality)
        assert len(self.x_dim) == self.n_modality  # check the length of x_dim

        # init networks
        self.net_q = [None]*self.n_modality
        self.net_p = [None]*self.n_modality
        for i in range(self.n_modality):
            self.net_q[i] = Q_net(self.net_N, self.x_dim[i], self.z_dim, self.p_drop)
            self.net_p[i] = P_net(self.net_N, self.x_dim[i], self.z_dim, self.p_drop)

        self.net_c = C_net(self.net_N, self.z_dim, self.n_classes, self.p_drop)
        self._nets = self.net_q + self.net_p + [self.net_c]
        # optimizers
        self.optimizer_q = [None]*self.n_modality
        self.optimizer_p = [None]*self.n_modality

        for i in range(self.n_modality):
            self.optimizer_q[i] = self.optimizer(
                self.net_q[i].parameters(),
                lr=opt.lr,
                **self.optimizer_params
            )
            self.optimizer_p[i] = self.optimizer(
                self.net_p[i].parameters(),
                lr=opt.lr,
                **self.optimizer_params
            )     
        self.optimizer_c = self.optimizer(
            self.net_c.parameters(),
            lr=opt.lr,
            **self.optimizer_params
        )
        self._optimizers = self.optimizer_q + self.optimizer_p + [self.optimizer_c]
        # scheduler
        self.scheduler_q = [None]*self.n_modality
        self.scheduler_p = [None]*self.n_modality
        for i in range(self.n_modality):
            self.scheduler_q[i] = self.scheduler(
                self.optimizer_q[i],
                **self.scheduler_params
            )
            self.scheduler_p[i] = self.scheduler(
                self.optimizer_p[i],
                **self.scheduler_params
            )
        self.scheduler_c = self.scheduler(
            self.optimizer_c,
            **self.scheduler_params
        )
        self._schedulers = self.scheduler_q + self.scheduler_p + [self.scheduler_c]

        # general
        self.opt =opt  
        self._metrics = ['loss_mse', 'loss_mse_cross', 'loss_survival', 'c_index'] # log the autoencoder loss and the classification loss
        if opt.log_time:
            self._metrics += ['t_mse', 't_mse_cross', 't_survival']

        # init variables
        #self.input_names = ['features', 'labels'] 
        self.init_vars(add_path=True)
        
        # init weights
        for i in range(self.n_modality):
            self.init_weight(self.net_q[i])
            self.init_weight(self.net_p[i])
        self.init_weight(self.net_c)

    def __str__(self):
        s = "OS CrossAutoencoder"
        return s
    
    def _train_on_batch(self, epoch, batch_idx, batch):
        net_q, net_p, net_c = self.net_q, self.net_p, self.net_c
        opt_q, opt_p, opt_c = self.optimizer_q, self.optimizer_p, self.optimizer_c
        for i in range(self.n_modality):
            net_q[i].train()
            net_p[i].train()
        net_c.train()

        X_list = [None]*self.n_modality
        X_recon_list = [None]*self.n_modality
        z_list = [None]*self.n_modality
        loss_mse_list = [None]*self.n_modality
        loss_mse_cross_list = []

        # read in training data 
        for i in range(self.n_modality):
            X_list[i] = batch[self.modality[i]]
        X_list = [tmp.cuda() for tmp in X_list]
        survival_time = batch['days'].cuda()
        survival_event = batch['event'].cuda()
        batch_size = X_list[0].shape[0]
        batch_log = {'size': batch_size}


        ###################################################
        # Stage 1: train Q and P with reconstruction loss #
        ###################################################
        for i in range(self.n_modality):
            net_q[i].zero_grad()
            net_p[i].zero_grad()
            for p in net_q[i].parameters():
                p.requires_grad=True
            for p in net_p[i].parameters():
                p.requires_grad=True
        for p in net_c.parameters():
            p.requires_grad=False

        t0 = time()
        for i in range(self.n_modality):
            z_list[i] = net_q[i](X_list[i])
            X_recon_list[i] = net_p[i](z_list[i])
            loss_mse = nn.functional.mse_loss(X_recon_list[i], X_list[i]) # Mean square error
            loss_mse_list[i] = loss_mse.item()
            loss_mse.backward()
            opt_q[i].step()
            opt_p[i].step()
        
        t_mse = time() - t0
        batch_log['loss_mse'] = sum(loss_mse_list)        
 
        #########################################################
        # Stage 2: train Q and P with cross reconstruction loss #
        #########################################################
        for i in range(self.n_modality):
            net_q[i].zero_grad()
            net_p[i].zero_grad()
            for p in net_q[i].parameters():
                p.requires_grad=True
            for p in net_p[i].parameters():
                p.requires_grad=True
        for p in net_c.parameters():
            p.requires_grad=False

        t0 = time()
        for i in range(self.n_modality):
            for j in range(i+1, self.n_modality):
                z_list[i] = net_q[i](X_list[i])
                z_list[j] = net_q[j](X_list[j])
                # Cross reconstruction
                X_recon_list[i] = net_p[i](z_list[j])
                X_recon_list[j] = net_p[j](z_list[i])
                loss_mse_j_i = nn.functional.mse_loss(X_recon_list[i], X_list[i]) # Mean square error
                loss_mse_i_j = nn.functional.mse_loss(X_recon_list[j], X_list[j]) # Mean square error
                loss_mse_cross = loss_mse_j_i + loss_mse_i_j
                loss_mse_cross_list.append(loss_mse_cross.item())
                loss_mse_cross.backward()
                opt_q[i].step()
                opt_p[i].step()
                opt_q[j].step()
                opt_p[j].step()    
        
        t_mse_cross = time() - t0
        batch_log['loss_mse_cross'] = sum(loss_mse_list)        

        ####################################################
        # Stage 3: train Q and C with classification loss  #
        ####################################################         
        if not survival_event.sum(0):  # skip the batch if all instances are negative
            batch_log['loss_survival'] = torch.Tensor([float('nan')])
            return batch_log
        for i in range(self.n_modality):
            net_q[i].zero_grad()
            for p in net_q[i].parameters():
                p.requires_grad=True
            for p in net_p[i].parameters():
                p.requires_grad=False
        net_c.zero_grad()
        for p in net_c.parameters():
            p.requires_grad=True

        t0 = time()
        for i in range(self.n_modality):
            z_list[i] = net_q[i](X_list[i])
        z_combined = torch.mean(torch.stack(z_list), dim=0) # get the mean of z_list
        pred = net_c(z_combined)
        loss_survival = neg_par_log_likelihood(pred, survival_time, survival_event)
        loss_survival.backward()
        for i in range(self.n_modality):
            opt_q[i].step()
        opt_c.step()
        t_survival = time() - t0
        c_index = CIndex(pred, survival_event, survival_time)
        batch_log['loss_survival'] = loss_survival.item()
        batch_log['c_index'] = c_index

        if self.opt.log_time:
            batch_log['t_mse'] = t_mse
            batch_log['t_mse_cross'] = t_mse_cross
            batch_log['t_survival'] = t_survival
        return batch_log

    def _vali_on_batch(self, epoch, batch_idx, batch):
        for i in range(self.n_modality):
            self.net_q[i].eval()
        self.net_c.eval()

        X_list = [None]*self.n_modality
        z_list = [None]*self.n_modality

        # read in training data
        for i in range(self.n_modality):
            X_list[i] = batch[self.modality[i]]
        X_list = [tmp.cuda() for tmp in X_list]
        survival_time = batch['days'].cuda()
        survival_event = batch['event'].cuda()
        batch_size = X_list[0].shape[0]
        batch_log = {'size': batch_size}
        if not survival_event.sum(0): # skip the batch if all instances are negative
            batch_log['loss_survival'] = torch.Tensor([float('nan')])
            return batch_log
        with torch.no_grad():
            for i in range(self.n_modality):
                z_list[i] = self.net_q[i](X_list[i])
            z_combined = torch.mean(torch.stack(z_list), dim=0) # get the mean of z_list
            pred = self.net_c(z_combined)
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
        return "Testing OS CrossAutoencder"
    
    def test_on_batch(self, batch_ind, batch):
        outdir = os.path.join(self.output_dir, 'batch%04d' % batch_ind)
        os.makedirs(outdir, exist_ok=True)

        for i in range(self.n_modality):
            self.net_q[i].eval()
        self.net_c.eval()

        X_list = [None]*self.n_modality
        z_list = [None]*self.n_modality
        # read in training data
        for i in range(self.n_modality):
            X_list[i] = batch[self.modality[i]]
        X_list = [tmp.cuda() for tmp in X_list]
        survival_time = batch['days'].cuda()
        survival_event = batch['event'].cuda()

        with torch.no_grad():
            for i in range(self.n_modality):
                z_list[i] = self.net_q[i](X_list[i])
            z_combined = torch.mean(torch.stack(z_list), dim=0) # get the mean of z_list
            pred = self.net_c(z_combined)

        loss = neg_par_log_likelihood(pred, survival_time, survival_event)
        print('loss', loss.item())
        c_index = CIndex(pred, survival_event, survival_time)
        print('c_index', c_index)
        eva = {'c_index': c_index}
        formatTable(eva, outpath=os.path.join(outdir, 'eva.csv'))
        output = self.pack_output(pred, batch, z_list)
        np.savez(os.path.join(outdir, 'batch%04d.npz' % batch_ind), **output)

    def pack_output(self, pred, batch, z_list):
        out = {}
        out['pred'] = pred.detach().cpu().numpy()
        out['event'] = batch['event'].cpu().numpy()
        out['days'] = batch['days'].cpu().numpy()
        for i in range(self.n_modality):
            z_list[i] = z_list[i].cpu().numpy()
        out['z_list'] = z_list
        return out
