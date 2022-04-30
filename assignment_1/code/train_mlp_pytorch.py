"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  predicted_class = predictions.max(1)[1]
  accuracy = torch.mean((predicted_class == targets).to(torch.float))

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  def loss_and_accuracy_curves(steps, loss, accuracy, min_acc):
    plt.title("Loss and accuracy curves")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.plot(steps,loss,label="Loss")
    plt.plot(steps,accuracy,label="Accuracy")
    plt.plot([0,FLAGS.max_steps],[min_acc,min_acc],'k--',label="Minimum accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig("mlp_pytorch.png")
    return

  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

  # Set device to cuda if cuda is available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  print("Device:",device)

  # Load test data
  x_test,t_test = cifar10["test"].images, cifar10["test"].labels
  x_test = np.resize(x_test,(x_test.shape[0],3*32*32))

  test_inputs = Variable(torch.from_numpy(x_test).to(device))
  test_labels = Variable(torch.from_numpy(t_test).to(device,torch.long).max(1)[1])

  # Initialize MLP
  mlp = MLP(3*32*32,dnn_hidden_units,10)

  # Initialize cross entropy module
  loss_module = nn.CrossEntropyLoss()

  # Initialize optimizer
  optimizer = optim.Adam(mlp.parameters(), lr=FLAGS.learning_rate, betas=(0.75,0.999), weight_decay=0.01)

  # Default optimizer for sanity check
  #optimizer = optim.SGD(mlp.parameters(), lr=FLAGS.learning_rate)

  if torch.cuda.is_available():
    mlp.cuda()

  steps = []
  losses = []
  accuracies = []
  for step in range(1,FLAGS.max_steps+1):
    x,y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = np.resize(x,(FLAGS.batch_size,3*32*32))

    inputs = Variable(torch.from_numpy(x).to(device))
    labels = Variable(torch.from_numpy(y).to(device,torch.long).max(1)[1])

    # Zero the optimizer gradients
    optimizer.zero_grad()

    # Forward pass
    out = mlp(inputs)

    # Loss on output
    loss = loss_module(out,labels)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()
 
    # Evaluate the model
    if step % FLAGS.eval_freq == 0:
      with torch.no_grad():
        predictions = mlp(test_inputs)

        loss = loss_module(predictions,test_labels)
        test_accuracy = accuracy(F.softmax(predictions,dim=1),test_labels)

      print("Step: %i, Loss: %f, Accuracy: %f" % (step,loss,test_accuracy))

      steps.append(step)
      losses.append(loss)
      accuracies.append(test_accuracy)

  loss_and_accuracy_curves(steps,losses,accuracies,0.53)

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
