# MIT License
#
# Copyright (c) 2017 Tom Runia
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


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, dropout_keep_prob,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size

        # Parameter for one-hot representation
        self.eye = torch.eye(vocabulary_size).to(device)

        # Embedding layer
        self.embedding = nn.Embedding(vocabulary_size,vocabulary_size)
        #self.embedding.weight.data = torch.eye(vocabulary_size)

        # Dropout layer
        self.dropout = nn.Dropout(p=1-dropout_keep_prob)

        # Layered LSTM
        self.lstm = nn.LSTM(vocabulary_size,lstm_num_hidden,lstm_num_layers)

        # Linear output module
        self.lin_out = nn.Linear(lstm_num_hidden,vocabulary_size)

        self.to(device)

    def forward(self, x, use_states=False):
        # Transform into one-hot vectors
        #out = F.embedding(x,self.eye)

        # Add embedding layer starting from one-hot encoding
        out = self.embedding(x)

        # LSTM forward
        if use_states:
            out, (self.hn,self.cn) = self.lstm(out, (self.hn,self.cn))
        else:
            out, (self.hn,self.cn) = self.lstm(out)

        # Dropout layer after LSTM
        out = self.dropout(out)

        # Transform output to vocabulary size
        out = self.lin_out(out).transpose(0,1).transpose(1,2)

        # Dropout layer after linear layer
        #out = self.dropout(out)

        return out
