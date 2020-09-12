#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import importlib
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,model1,model2):
        super(Model,self).__init__()
        self.model_x = model1
        self.model_delta_x = model2
        # for exchange_rate
        dim = 8
        dim1 = 8
        # for energy
        # dim = 26
        # dim1 = 26
        # for nasdaq
        # dim = 82
        # dim1 = 82
        # for solar_energy
        # dim = 137
        # dim1 = 137
        self.T = nn.Sequential(
                            nn.Linear(2*dim,dim),
                            # nn.Linear(dim,dim),
                            nn.Sigmoid()
                            # nn.ReLU(),
                            # nn.Linear(dim,dim),
                            # nn.Sigmoid()
                            )
        self.S = nn.Sequential(
                            nn.Linear(dim,dim1),
                            # nn.Sigmoid(),
                            # nn.ReLU(),
                            nn.PReLU(1),
                            nn.Linear(dim1,dim),
                            nn.Sigmoid()
                            )

        
    def forward(self,x,delta_x):
        # print("x",x.shape)
        xt_pre = self.model_x(x) # highway CNN
        delta_x_pre1 = self.model_delta_x(delta_x)
        # print("delta_x_pre1",delta_x_pre1.shape)
        xt = self.S(x[:,-1,:].reshape(x.shape[0],-1))
        # # print("xt",xt.shape)
        Tx = self.T(torch.cat([delta_x_pre1,xt],1))
        # # print("Tx",Tx.shape)
        delta_x_pre = Tx.mul(xt)+(1-Tx).mul(delta_x_pre1)

        # print("delta_x_pre",delta_x_pre.shape)
        # delta_x_pre = delta_x_pre1
        
        return xt_pre,delta_x_pre,(xt_pre+delta_x_pre)
        # return xt_pre,delta_x_pre1,(xt_pre+delta_x_pre1)