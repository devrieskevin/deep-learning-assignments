"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    
    self.params['weight'] = np.random.normal(0.0,0.0001,(out_features,in_features))
    self.params['bias'] = np.zeros(out_features)

    self.grads['weight'] = np.zeros((out_features,in_features))
    self.grads['bias'] = np.zeros(out_features)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("Forward on Linear",x.shape)

    out = np.einsum('ij,bj->bi',self.params['weight'],x) + self.params['bias'][None,:]

    # Store input for backward pass
    self.x_input = x

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("Backward on Linear",dout.shape)
    #print(np.sum(dout))

    # Calculate parameter gradients
    self.grads['weight'] = np.einsum('bj,bk->jk',dout,self.x_input)
    self.grads['bias'] = np.einsum('bj->j',dout)

    dx = np.einsum('ij,bi->bj',self.params['weight'],dout)

    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("Forward on ReLU")

    # Store input for backward pass
    self.x_input = x
    
    out = np.maximum(0,x)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("Backward on ReLU")
    #print(np.sum(dout))

    # Alternative solution
    #dx = dout
    #dx[self.x_input <= 0] = 0

    dx = (self.x_input > 0) * dout

    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("Forward on SoftMax")

    b = np.amax(x,axis=1)[:,None]
    y = np.exp(x - b)
    out = y / np.sum(y,axis=1)[:,None]

    # Store softmax for backward pass
    self.softmax = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("Backward on SoftMax")
    #print(np.sum(dout))

    # Einsum method
    #classes = self.softmax.shape[1]
    #grad = np.einsum("bi,ij->bij",self.softmax,np.eye(classes))
    #grad = grad - np.einsum("bi,bj->bij",self.softmax,self.softmax)
    #dx = (grad @ dout[:,:,None])[:,:,0]

    # Matrix multiplication method
    temp = self.softmax[:,:,None] @ self.softmax[:,None,:]
    dx = (dout * self.softmax) - np.einsum('bij,bi->bj',temp,dout)

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("Forward on Cross-Entropy")

    batch_size = x.shape[0]

    # Argmax solution
    #y_argmax = np.argmax(y,axis=1)
    #out = -np.sum(np.log(x[np.arange(x.shape[0]),y_argmax])) / batch_size

    # Sum solution
    out = -np.sum(y*np.log(x)) / batch_size

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("Backward on Cross-Entropy")

    batch_size = x.shape[0]

    # Builds Jacobian
    #dx = np.zeros(y.shape)
    #y_argmax = np.argmax(y,axis=1)
    #dx[np.arange(x.shape[0]),y_argmax] = -1 / x[np.arange(x.shape[0]),y_argmax]

    dx = -(y / x) / batch_size

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
