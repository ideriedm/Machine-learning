# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import matplotlib.pyplot as plt
import numpy as np


def compute_loss(y, tx, w):
    
  # compute loss by MSE
    e = y - np.dot(tx,w)
    loss = 1/(2*e.shape[0])*(e**2).sum(axis = 0)
    
    return loss



def compute_gradient(y, tx, w):
    """Compute the gradient."""
    #  compute gradient and error vector
    e = y - np.dot(tx,w)
    gradient = -1/y.shape[0]*np.dot(np.transpose(tx),e)
    
    return gradient



def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    #implement stochastic gradient computation.
    e = y - np.dot(tx,w)
    gradient = -1/y.shape[0]*np.dot(np.transpose(tx),e)
    
    return gradient, e




def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    """Linear regression using gradient descent
        Should return : (w,loss) 
        [last w vector + corresponding loss]
    """
   ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w = w - gamma*gradient
        ws.append(w)
        losses.append(loss)
        
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
    
    
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
        Should return : (w,loss) 
        [last w vector + corresponding loss]
    """
    # Linear regression using stochastic gradient descent
    ws = [initial_w]
    losses = []
    w = intial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches = 1):
            #compute a stochastic gradient and loss
            gradient, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            #update w through the stochastic gradient update
            w = w - gamma * gradient
            #calculate loss
            loss = compute_loss(y, tx, w)
            #store w and loss
            ws.append(w)
            losses.append(loss)
    
        print("SGD({bi}/{tri}): loss = {l}, w0 ={w0}, w1 ={w1}".format(bi = n_iter, tri=max_iters-1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
    

    
def least_squares(y, tx):
    """Least squares regression using normal equations
        Should return : (w,loss) 
        [last w vector + corresponding loss]
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: Least squares regression using normal equations
    # ***************************************************
    
    raise NotImplementedError
    
    
def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations
        Should return : (w,loss) 
        [last w vector + corresponding loss]
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: Ridge regression using normal equations
    # ***************************************************
    
    raise NotImplementedError
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD
        Should return : (w,loss) 
        [last w vector + corresponding loss]
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: Logistic regression using gradient descent or SGD
    # ***************************************************
    
    raise NotImplementedError
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD
        Should return : (w,loss) 
        [last w vector + corresponding loss]
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: Regularized logistic regression using gradient descent or SGD
    # ***************************************************
    
    raise NotImplementedError
    
    
    
    
    
    
    
