import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  
  # Add regularization to the loss.
  loss /= num_train
  dW /= num_train # as if loss becomes loss/3 then dloss/dW also becomes (dloss/dw)/3
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #wX  num_train X C
  scores = X.dot(W)

  # w_y.X (scores of correct class for all X)
  num_train = X.shape[0]

  correct_class_score = scores[np.arange(num_train), y] #shape: (num_train,)

  # reshaping for broadcasting
  correct_class_score = correct_class_score.reshape((num_train, 1))

  margin_matrix = np.maximum(0, scores - correct_class_score + 1)

  #margin matrix for the correct class should be 0 but currently it is 1 due to delta
  margin_matrix[np.arange(num_train), y] = 0

  loss = np.sum(margin_matrix)

  loss += reg*np.sum((W*W))

  loss /= num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # to keep track of how many values are contributing to gradients we need to find 
  # all non zero values in margin matrix
  margin_matrix_ = margin_matrix
  margin_matrix_[margin_matrix > 0] = 1  # rest will be 0

  # now if there are A classes in a row(X) contributing to total (non zero) 
  # loss, the correct class for that X will contribute -A*(X) towards the final gradient
  # so finding A by sum

  row_sum = np.sum(margin_matrix_, axis = 1)

  #combining contributions to gradient by correct classes (just coefficients)
  margin_matrix_[np.arange(num_train), y] = -row_sum.T
  
  #finding gradient
  dW = np.dot(X.T, margin_matrix_) 
  dW += 2*reg*W
  dW /= num_train
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
