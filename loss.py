#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

import torch
from torch import nn
import math

class CWAUCHLoss(nn.Module):
    def __init__(self, alpha=1,lamb=1, num_hard = 0):
        super(CWAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = CWCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.alpha = alpha
        self.lamb = lamb
        print(alpha,lamb)

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # import pdb;pdb.set_trace()
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                print("neg")
            else:
                trans_pos = out_pos.repeat(num_neg,1)
                trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
        except:
            import pdb;pdb.set_trace()

        cls = self.classify_loss(outs,labels)[0] + self.alpha * penalty_term
        # import pdb;pdb.set_trace()
        return cls, self.alpha * penalty_term
      
class CWCELoss(nn.Module):
    def __init__(self, num_hard=0):
        super(CWCELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
        self.epsilon = 1e-32
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        num_neg = neg_labels.sum()
        num_pos = labels.sum()
        
        Beta_P = num_pos / (num_pos + num_neg)
        Beta_N = num_neg / (num_pos + num_neg)
        # import pdb;pdb.set_trace()
        
        pos_loss = torch.mul(labels,torch.log(outs+ self.epsilon))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs+ self.epsilon))
        fpcls = - Beta_N * pos_loss.mean() - Beta_P * neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
        return fpcls , fpcls 
