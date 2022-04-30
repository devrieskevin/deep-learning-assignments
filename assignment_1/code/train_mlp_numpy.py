"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

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

  predicted_class = np.argmax(predictions,axis=1)
  target_class = np.argmax(targets,axis=1)

  accuracy = np.mean(predicted_class == target_class)

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
    plt.savefig("mlp_numpy.png")
    return

  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

  # Load test data
  x_test,t_test = cifar10["test"].images, cifar10["test"].labels
  x_test = np.resize(x_test,(x_test.shape[0],3*32*32))

  # Initialize MLP
  mlp = MLP(3*32*32,dnn_hidden_units,10)

  # Initialize cross entropy module
  cross_entropy_module = CrossEntropyModule()

  steps = []
  losses = []
  accuracies = []
  for step in range(1,FLAGS.max_steps+1):
    x,y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = np.resize(x,(FLAGS.batch_size,3*32*32))
    #x -= np.mean(x,axis=1)[:,None]
    
    # Forward pass
    out = mlp.forward(x)

    #print("Intermediate Loss: %f" % (cross_entropy_module.forward(out,y)))
    #print(out)

    # Loss gradient on output
    dout = cross_entropy_module.backward(out,y)

    # Backward pass
    mlp.backward(dout)

    # Update parameters
    for n in range(len(mlp.linear_modules)):
      #print(np.sum(mlp.linear_modules[n].params["weight"]))

      mlp.linear_modules[n].params['weight'] -= FLAGS.learning_rate * mlp.linear_modules[n].grads['weight']
      mlp.linear_modules[n].params['bias'] -= FLAGS.learning_rate * mlp.linear_modules[n].grads['bias']

      #print(np.sum(mlp.linear_modules[n].params["weight"]))

    # Evaluate the model
    if step % FLAGS.eval_freq == 0:
      predictions = mlp.forward(x_test)

      loss = cross_entropy_module.forward(predictions,t_test)
      test_accuracy = accuracy(predictions,t_test)

      print("Step: %i, Loss: %f, Accuracy: %f" % (step,loss,test_accuracy))

      steps.append(step)
      losses.append(loss)
      accuracies.append(test_accuracy)

  loss_and_accuracy_curves(steps,losses,accuracies,0.46)

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
