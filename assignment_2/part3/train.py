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

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file,config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize model
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, \
                                config.dropout_keep_prob, \
                                config.lstm_num_hidden, config.lstm_num_layers, device)

    # Initialize optimizer
    optimizer = torch.optim.RMSprop(model.parameters(),lr=config.learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.learning_rate_step, 
                                                gamma=config.learning_rate_decay)

    # Initialize or load the model that we are going to use
    if config.model_state and config.optimizer_state and \
       config.scheduler_state and config.load_state:

        print("Loading model state")
        model.load_state_dict(torch.load(config.model_state))
        optimizer.load_state_dict(torch.load(config.optimizer_state))
        scheduler.load_state_dict(torch.load(config.scheduler_state))

    # Setup the loss
    criterion = torch.nn.CrossEntropyLoss()

    print("Training model on:", device)

    for epoch in range(1,config.epochs+1):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            # Advance scheduler
            scheduler.step()

            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_targets = torch.stack(batch_targets).t().to(device)

            # Set grads to zero
            optimizer.zero_grad()

            # Compute loss and grads
            out = model(batch_inputs)
            #loss = criterion(out,batch_targets)

            # Explicit averaging over sequence losses
            loss = 0.0
            for t in range(config.seq_length):
                loss += criterion(out[:,:,t],batch_targets[:,t])
            loss /= config.seq_length

            loss.backward()

            # Clip gradient norm to prevent exploding gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

            # Update parameters
            optimizer.step()

            # Check accuracy over current batch
            accuracy = (out.max(dim = 1)[1] == batch_targets).to(torch.float).mean()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:
                print("[{}] Epoch {}, Train Step {:04d}/{:04d}, Batch Size = {}, "
                      "Examples/Sec = {:.2f}, Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch, step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                sentence = torch.randint(dataset.vocab_size,(1,)).to(device,torch.long)

                with torch.no_grad():
                    # Set model in evaluation mode
                    model.eval()

                    for t in range(1,config.example_len):
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

                    # Set model back into training mode
                    model.train()

                print("Example sentence:",dataset.convert_to_string(sentence.cpu().numpy()))

                # Save the model after generating a sentence
                if config.model_state and config.optimizer_state and \
                   config.scheduler_state and config.save_state:

                    print("Saving model state")
                    torch.save(model.state_dict(),config.model_state)
                    torch.save(optimizer.state_dict(),config.optimizer_state)
                    torch.save(scheduler.state_dict(),config.scheduler_state)

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                print("Max Train step reached")
                break

    print('Done training.')


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

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    # Student specified parameters
    parser.add_argument('--model_state',type=str, default="model_state.pt", \
                        help="Path to a file containing the model state")

    parser.add_argument('--optimizer_state',type=str, default="optimizer_state.pt", \
                        help="Path to a file containing the optimizer state")

    parser.add_argument('--scheduler_state',type=str, default="scheduler_state.pt", \
                        help="Path to a file containing the optimizer state")

    parser.add_argument('--save_state', default=False, action='store_true', help="Saves model state")
    parser.add_argument('--load_state', default=False, action='store_true', help="Loads model state")
    parser.add_argument('--epochs', type=int, default=10,help="Number of epochs to train")
    parser.add_argument('--example_len', type=int, default=30,help="Length of example sentences to generate")
    parser.add_argument('--temperature', type=float, default=0.5,help="Temperature variable for character sampling")

    config = parser.parse_args()

    # Train the model
    train(config)
