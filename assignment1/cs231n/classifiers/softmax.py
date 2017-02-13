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
  
  # Learning rate 
  vlambda = 0.5
  
  # Check matrices shape and calculate the score (f)
  mw, nw = np.shape(W)
  mx, nx = np.shape(X)
  if nw == mx:
      pass
  
  elif mw == nx: # Matrice are transposed
      W = np.transpose(W)
      X = np.transpose(X)
      y = np.transpose(y)
      dW = np.zeros_like(W)
      
  # Get shapes
  num_classes = np.shape(W)[0]
  num_train = np.shape(X)[1]
  
  # Loop for calculate each score for each class
  for i in xrange(num_train):
    
      # Score calculation 
      f_j = np.dot(W, X[:,i]) # Score fj in the equations

      # Regularization -> log (C) = max(f) -> fj = fj - log(C)
      log_C = np.max(f_j)
      f_j -= log_C
      
      # Calculate f_yi
      f_yi = f_j[y[i]]
      
      # Calculate loss -> L_i = -f_yi + ln(sumatoria(e^f_j))
      loss += -f_yi + np.log(np.sum(np.exp(f_j)))
      
      # calculate the probabilities that the sample belongs to each class
      probabilities = np.exp(f_j) / np.sum(np.exp(f_j))
      probabilities[y[i]] -= 1 # calculate p-1 and later we'll put the negative back

      # Compute gradient:
      # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
      
      # Loop in classes
      for j in xrange(num_classes):
          # Probability calculation
          # Use f_j[clase] because it is calculating all probabilities
          # p = np.exp(f_yi) / np.sum(np.exp(f_j))
          dW[j,:] += probabilities[j] * X [:, i]
          # dW[clase,:] += (p - (clase != y[clase])) * X [:, i]

  # Compute the average
  loss /= num_train
  dW /= num_train
  
  # Regularization
  loss += vlambda * reg * np.sum(np.square(W))
  dW += reg*W 
  
  # Change dimensions of dW because it is transposed 
  dW = np.transpose(dW)
  
  pass
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
  
  # ------------------------- LOSS ----------------------- #
  # Learning rate 
  vlambda = 0.5
  
  # Check matrices shape and calculate the score (f)
  mw, nw = np.shape(W)
  mx, nx = np.shape(X)
  if nw == mx:
      pass
  
  elif mw == nx: # Matrice are transposed
      W = np.transpose(W)
      X = np.transpose(X)
      dW = np.zeros_like(W)
      
  # Get shapes
  num_train = np.shape(X)[1]
  
  # Score calculation -> f_j (C, N)
  f_j = np.dot(W,X)
  # Regularization -> log (C) = max(f) -> fj = fj - log(C)
  log_C = np.max(f_j, axis = 0, keepdims = True)
  #log_C = np.array(log_C).reshape((np.shape(log_C)[0], 1))
  f_j -= log_C
  # Calculate probabilities
  probabilities = np.exp(f_j) / np.sum(np.exp(f_j), axis = 0, keepdims = True)
             
  # Calculate loss -> L_i = -ln(e^fyi/sum(e^fj)) -> probabilities of each class 
  loss = np.sum(-np.log(probabilities[y, np.arange(num_train)]))
  
  # ----------------------- SGD -------------------- #
  # Compute gradient:
  # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
  
  # Gradient calculation
  probabilities[y,np.arange(num_train)] -= 1
  dW = np.dot(probabilities, np.transpose(X))  
  
  # Compute the average
  loss /= num_train
  dW /= num_train
  
  # Regularization
  loss += vlambda * reg * np.sum(np.square(W))
  dW += reg*W 
  
  # Change dimensions of dW because it is transposed 
  dW = np.transpose(dW)
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

