import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N,D = X.shape
  scores = np.zeros([N,W.shape[1]])
  for i in range(N):
      weighted_output = np.dot(X[i],W)
      total = 0.0; maxval=np.max(weighted_output)
      for j in range(W.shape[1]):
          scores[i,j] = np.exp(weighted_output[j]-maxval)
          total += scores[i,j]
      scores[i] /= total
      loss += -np.log(scores[i,y[i]]+1e-10) # add tiny number to prevent overflow

      for j in range(W.shape[1]):
          if j==y[i]:
              dW[:,y[i]] += -(1.0 - scores[i,y[i]])*X[i].T
          else:
              dW[:,j] += scores[i,j]*X[i].T

  loss /= N
  loss += reg*np.sum(W*W)

  dW /= N
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  scores = X.dot(W)
  maxval = np.max(scores,axis=1,keepdims=True)
  temp = np.exp(scores-maxval)
  scores = temp/np.sum(temp,axis=1,keepdims=True)
  loss += np.sum(-np.log(scores[list(range(N)),y]))
  loss /= N
  loss += reg*np.sum(W*W)

  scores[list(range(N)),y] -= 1
  dW += X.T.dot(scores)
  dW /= N
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
