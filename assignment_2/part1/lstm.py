################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

		# Store important parameters for later
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.batch_size = batch_size

        normal = torch.distributions.normal.Normal(0,0.001)

        # Initialize hidden units at t = 0
        self.h_init = torch.zeros(num_hidden).to(device)
        self.c_init = torch.zeros(num_hidden).to(device)

        # Initialize weight matrices
        self.Wgh = nn.Parameter(normal.sample((num_hidden,num_hidden)))
        self.Wih = nn.Parameter(normal.sample((num_hidden,num_hidden)))
        self.Wfh = nn.Parameter(normal.sample((num_hidden,num_hidden)))
        self.Woh = nn.Parameter(normal.sample((num_hidden,num_hidden)))

        self.Wgx = nn.Parameter(normal.sample((num_hidden,input_dim)))
        self.Wix = nn.Parameter(normal.sample((num_hidden,input_dim)))
        self.Wfx = nn.Parameter(normal.sample((num_hidden,input_dim)))
        self.Wox = nn.Parameter(normal.sample((num_hidden,input_dim)))

        self.Wph = nn.Parameter(normal.sample((num_classes,num_hidden)))

        # Initialize biases
        self.bg = nn.Parameter(torch.zeros(num_hidden))
        self.bi = nn.Parameter(torch.zeros(num_hidden))
        self.bf = nn.Parameter(torch.zeros(num_hidden))
        self.bo = nn.Parameter(torch.zeros(num_hidden))

        self.bp = nn.Parameter(torch.zeros(num_classes))

        # Export model to device
        self.to(device)

    def forward(self, x):
        # Prepare start and end indices for each sequence
        seq_idx = torch.arange(0,self.input_dim*self.seq_length,self.input_dim).to(torch.long)

        h = self.h_init[None,:,None]
        c = self.c_init[None,:,None]
        for t in range(len(seq_idx)-1):
            start,end = seq_idx[t],seq_idx[t+1]

            g = F.tanh(self.Wgx @ x[:,start:end,None] + self.Wgh @ h + self.bg[None,:,None])
            i = F.sigmoid(self.Wix @ x[:,start:end,None] + self.Wih @ h + self.bi[None,:,None])
            f = F.sigmoid(self.Wfx @ x[:,start:end,None] + self.Wfh @ h + self.bf[None,:,None])
            o = F.sigmoid(self.Wox @ x[:,start:end,None] + self.Woh @ h + self.bo[None,:,None])
            c = g * i + c * f
            h = F.tanh(c) * o

        p = (self.Wph @ h + self.bp[None,:,None])[:,:,0]

        return p
