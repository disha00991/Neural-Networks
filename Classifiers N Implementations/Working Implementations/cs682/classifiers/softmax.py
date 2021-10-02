import numpy as np
from random import shuffle

# class Softmax(object):
#   """ a Softmax classifier """

#   def __init__(self):
#     self.W = np.random.randn(3073, 10) * 0.0001

#   def train(self, X, y, reg, lr, no_of_epochs):
#     self.X_train = X
#     self.y_train = y
#     for i in range(no_of_epochs):
#       loss_vectorized, dW = softmax_loss_vectorized(self.W, self.X_train, self.y_train, reg)
#       self.W -= lr * dW

  


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  for i in range(num_train):

    # numerators
    all_numerators = np.exp(X[i].dot(W))
    # denominator
    denom = np.sum(all_numerators)

    all_scores = all_numerators/denom

    correct_class_score = all_scores[y[i]]

    loss += -1*np.log(correct_class_score)
    
    #dW when found for dL/dw11 where w11 is the weight for dimnesion = 1 and class = 1
    for d in range(X.shape[1]): # dimensions
      for c in range(num_classes):
        if (c != y[i]):
          dW[d, c] += X[i, d] * all_scores[c]
        else:
          dW[d, c] += X[i, d] * (all_scores[c] - 1)   
    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += reg * np.sum(W*W)

  dW /= num_train
  dW += 2 * reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = np.exp(X.dot(W))
  #normalized scores (probability)
  scores = f/np.sum(f, axis=1).reshape((num_train, 1))

  correct_class_scores = scores[np.arange(num_train), y]

  loss = np.sum(-1*np.log(correct_class_scores))

  #dW
  score_coeff_matrix = scores
  score_coeff_matrix[np.arange(num_train), y] -= 1

  dW = X.T.dot(score_coeff_matrix)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= num_train
  loss += reg * np.sum(W*W)

  dW /= num_train
  dW += 2 * reg * W
  return loss, dW
