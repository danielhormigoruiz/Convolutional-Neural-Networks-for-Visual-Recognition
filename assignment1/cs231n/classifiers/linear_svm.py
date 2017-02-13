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
  delta = 1.0
  vlambda = 0.5
#  for i in xrange(num_train):
#    scores = X[i].dot(W)
#    correct_class_score = scores[y[i]]
#    for j in xrange(num_classes):
#      if j == y[i]:
#        continue
#      margin = scores[j] - correct_class_score + delta # note delta = 1
#      if margin > 0:
#        loss += margin
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i] 
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += vlambda * reg * np.sum(W * W)
  
  # Adding the gradient of regularization
  dW += reg * W
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
  delta = 1.0
  vlambda = 0.5
  num_train = np.shape(X)[0]
  
  # Check matrices shape and calculate the score
  mw, nw = np.shape(W)
  mx, nx = np.shape(X)
  if nw == mx:
      score = np.dot(W,X)
  elif mw == nx: # Matrice are transposed
      score = np.dot(np.transpose(W),np.transpose(X))
      
  # Get the scores of each class indicated in 'y'
  score_y = np.zeros(np.shape(score)[1])
  row = np.array(y)
  colum = np.arange(np.shape(y)[0])
  score_y = score[row, colum]
  # Calculate the Li
  margin = score - score_y + delta
  # In these positions in which magin[i,j] = y[i] -> margin = 0
  margin[row,colum] = 0.0

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss = np.sum(margin) 
  loss /= num_train
  # Add regularization to the loss.
  loss += vlambda * reg * np.sum(np.square(W))
    
  pass
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
  ds = np.zeros(margin.shape)
  ds[margin>0] = 1
  ds[y,range(num_train)] = -np.sum(margin>0,axis=0)
  dW = np.dot(X.T,ds.T)
  dW /= num_train  
  dW += reg*W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
