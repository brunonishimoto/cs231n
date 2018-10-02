import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)

    ## Calculating loss
    # This is a trick for avoiding numeric stability (large exponentiations)
    scores -= np.max(scores)

    # Exponential
    scores_exp = np.exp(scores)

    # Normalizing so the scores sums up to 1.
    sum = np.sum(scores_exp)
    scores_norm = scores_exp / sum

    # The loss for this training data is the negative log of it's score
    loss += -np.log(scores_norm[y[i]])

    ## Calculating gradient
    dlog     = -(1 / scores_norm[y[i]]) # gradient for -log
    ddiv_num = (1 / sum) # gradient for numerator of division
    ddiv_den = -(scores_exp[y[i]] / np.square(sum)) # gradient for denominator of division

    gradient_other_class = X[i][:, None] * dlog * ddiv_den * scores_exp[None, :]

    dW += gradient_other_class
    dW[:, y[i]] += dlog * ddiv_num * scores_exp[y[i]] * X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

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
  num_train = X.shape[0]

  # Scores
  scores = X.dot(W) # dim: (N, C)
  scores -= np.max(scores, axis=-1)[:, None]

  # Loss
  term1      = -scores[np.arange(num_train), y] # dim: (N, )
  scores_exp = np.exp(scores) # dim: (N, C)
  sums_j     = np.sum(scores_exp, axis=-1) # dim: (N, )
  term2      = np.log(sums_j) # dim: (N, )
  loss       = np.sum(term1 + term2)

  # Gradient
  coef = scores_exp / sums_j[:, None]
  coef[np.arange(num_train), y] -= 1
  dW = X.T.dot(coef)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

