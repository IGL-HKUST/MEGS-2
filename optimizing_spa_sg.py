# -*- coding:utf-8 -*-
# 
# Author: 
# Time: 

import torch
import fnmatch
import numpy as np
import os

class OptimizingSpaSG:
    def __init__(self, gaussians, opt, device,imp_score_flag = False):
        self.gaussians = gaussians
        self.device = device
        self.imp_score_flag=imp_score_flag
        self.init_rho = opt.rho_lr
        self.prune_ratio=opt.sharpness_ratio
        self.u = {}
        self.z = {}
        sharpness = self.gaussians.get_sg_sharpness
        self.u = torch.zeros(sharpness.shape).to(device)
        self.z = torch.Tensor(sharpness.data.cpu().clone().detach()).to(device)

    def update(self, imp_score, update_u= True):
        z = self.gaussians.get_sg_sharpness + self.u
        if self.imp_score_flag == True:
            self.z = torch.Tensor(self.prune_z_metrics_imp_score(z,imp_score)).to(self.device)
        else:
            self.z = torch.Tensor(self.prune_z(z)).to(self.device)
        if update_u:
            with torch.no_grad():
                diff =  self.gaussians.get_sg_sharpness  - self.z
                self.u += diff
                    
    def prune_z(self, z):
        z_flat = z.view(-1)
        index = int(self.prune_ratio * len(z_flat))
        z_sort = {}
        z_update = torch.zeros(z_flat.shape)
        z_sort, _ = torch.sort(z_flat, 0)
        z_threshold = z_sort[index-1]
        z_update = ((z_flat > z_threshold) * z_flat)
        z_update = z_update.view(z.shape)  
        return z_update

    def append_spa_loss_sg(self, loss):
        loss += 0.5 * self.init_rho * (torch.norm(self.gaussians.get_sg_sharpness - self.z + self.u, p=2)) ** 2
        return loss

    def adjust_rho(self, iteration, iterations, factor=5):
        if iteration > int(0.85 * iterations):
            self.rho = factor * self.init_rho
            
    def prune_z_metrics_imp_score(self, z, imp_score): 
        z_flat = z.view(-1)  
        imp_score_flat = imp_score.view(-1)  

        index = int(self.prune_ratio * len(z_flat))
        imp_score_sort, _ = torch.sort(imp_score_flat, 0)
        imp_score_threshold = imp_score_sort[index-1]
        indices = imp_score_flat < imp_score_threshold
        z_flat[indices] = 0

        z_result = z_flat.view(z.shape)
        return z_result
      