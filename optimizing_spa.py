# -*- coding:utf-8 -*-
# 
# Author: 
# Time: 

import torch
import fnmatch
import numpy as np
import os


class OptimizingSpa:
    def __init__(self, gaussians, opt, device,imp_score_flag = False):
        self.gaussians = gaussians
        self.device = device
        self.imp_score_flag=imp_score_flag
        self.init_rho = opt.rho_lr
        self.prune_ratio=opt.prune_ratio2
        self.u = {}
        self.z = {}
        opacity = self.gaussians.get_opacity
        self.u = torch.zeros(opacity.shape).to(device)
        self.z = torch.Tensor(opacity.data.cpu().clone().detach()).to(device)

    def update(self, imp_score, update_u= True):
        z = self.gaussians.get_opacity + self.u
        if self.imp_score_flag == True:
            self.z = torch.Tensor(self.prune_z_metrics_imp_score(z,imp_score)).to(self.device)
        else:
            self.z = torch.Tensor(self.prune_z(z)).to(self.device)
        if update_u:
            with torch.no_grad():
                diff =  self.gaussians.get_opacity  - self.z
                self.u += diff
                    
    def prune_z(self, z):
        index = int(self.prune_ratio * len(z))
        z_sort = {}
        z_update = torch.zeros(z.shape)
        z_sort, _ = torch.sort(z, 0)
        z_threshold = z_sort[index-1]
        z_update= ((z > z_threshold) * z)  
        return z_update

    def append_spa_loss(self, loss):
        loss += 0.5 * self.init_rho * (torch.norm(self.gaussians.get_opacity - self.z + self.u, p=2)) ** 2
        return loss

    def adjust_rho(self, iteration, iterations, factor=5):
        if iteration > int(0.85 * iterations):
            self.rho = factor * self.init_rho
            
    def prune_z_metrics_imp_score(self, z, imp_score):
        index = int(self.prune_ratio * len(z))
        imp_score_sort = {}
        imp_score_sort, _ = torch.sort(imp_score, 0)
        imp_score_threshold = imp_score_sort[index-1]
        indices = imp_score < imp_score_threshold 
        z[indices == 1] = 0  
        return z  
      


