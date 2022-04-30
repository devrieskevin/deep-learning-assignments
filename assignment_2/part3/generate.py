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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def generate(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset
    dataset = TextDataset(config.txt_file,config.seq_length)

    # Initialize model
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, \
                                config.dropout_keep_prob, \
                                config.lstm_num_hidden, config.lstm_num_layers, device)


    print("Loading model state")
    model.load_state_dict(torch.load(config.model_state))

    # Export to the device from cpu
    model = model.to(device)

    # Set model in evaluations mode
    model.eval()

    if config.input_sentence:
        sentence = torch.Tensor([dataset._char_to_ix[char] for char in config.input_sentence]).to(device,torch.long)
    else:
        sentence = torch.randint(dataset.vocab_size,(1,)).to(device,torch.long)

    with torch.no_grad():

        for t in range(len(sentence),config.example_len):
            if t <= config.seq_length:
                pred = model(sentence[:,None],use_states=False)
            else:
                pred = model(sentence[t-config.seq_length:t,None],use_states=True)

            # Random sampling
            probs = F.softmax(pred / config.temperature,dim=1)[0,:,-1]
            distribution = torch.distributions.Categorical(probs)
            next_char = distribution.sample((1,)).to(device)

            # Greedy sampling
            #next_char = (pred.max(dim = 1)[1])[:,-1]

            sentence = torch.cat((sentence,next_char))

    print("Generated sentence:")
    print(dataset.convert_to_string(sentence.cpu().numpy()))


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    # Student specified parameters
    parser.add_argument('--model_state',type=str, default="model_state.pt", \
                        help="Path to a file containing the model state")

    parser.add_argument('--example_len', type=int, default=30,help="Length of example sentences to generate")
    parser.add_argument('--input_sentence', type=str, default=None,help="Input sentence to complete")

    parser.add_argument('--temperature', type=float, default=0.5,help="Temperature variable for character sampling")

    config = parser.parse_args()

    # Generate a sentence from the model
    generate(config)
