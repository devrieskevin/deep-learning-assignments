"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    n_nums = [n_inputs] + n_hidden + [n_classes]

    # Initialize modules
    self.linear_modules = [LinearModule(n_nums[n-1],n_nums[n]) for n in range(1,len(n_nums))]
    self.relu_modules = [ReLUModule() for n in range(len(self.linear_modules)-1)]
    self.softmax_module = SoftMaxModule()

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    out = x

    # Forward passing of subsequent linear modules followed by activations
    for n in range(len(self.linear_modules)-1):
        out = self.relu_modules[n].forward(self.linear_modules[n].forward(out))

    out = self.softmax_module.forward(self.linear_modules[-1].forward(out))

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    dx = self.linear_modules[-1].backward(self.softmax_module.backward(dout))

    # Backward pass through all linear modules and activations
    for n in np.arange(len(self.linear_modules)-1)[::-1]:
        dx = self.linear_modules[n].backward(self.relu_modules[n].backward(dx))

    ########################
    # END OF YOUR CODE    #
    #######################

    return
