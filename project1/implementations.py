# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import matplotlib.pyplot as plt
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def MAE(error):
    """Calculate the MAE of the error vector"""

    return np.mean(np.abs(error))

def MSE(error):
    """Calculate the MSE of the error vector"""

    return 1/2*np.mean(error**2)

def RMSE(mse):
    """Calculate the RMSE of the MSE"""

    return np.sqrt(2*mse)

def compute_gradient(y, tx, w):
    """Compute the gradient."""

    # compute gradient and error vector
    # For MSE, ▽L = -1/N*X_T*e
    error = y - np.dot(tx,w)

    # tx.T = tx.transpose()
    return -np.dot(tx.T,error)/len(y), error

def compute_loss_MSE(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    error = y - np.dot(tx,w)
    # or y - tx @ w
    # or y - tx.dot(w)

    return MSE(error)

def compute_loss_MAE(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    error = y - np.dot(tx,w)
    # or y - tx @ w
    # or y - tx.dot(w)

    return MAE(error)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    """Linear regression using gradient descent
        Should return : (w,loss)
        [last w vector + corresponding loss]
    """

    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient, error = compute_gradient(y, tx, w)
        loss = MSE(error)
        # or loss = MAE(error)

        # update w by gradient
        w = w - gamma * gradient

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w

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

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # It's same as the gradient descent.

    return compute_gradient(y, tx, w)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
        Should return : (w,loss)
        [last w vector + corresponding loss]
    """

    # Define parameter w
    w = initial_w

    for n_iter in range(max_iters):
        for y_,tx_ in batch_iter(y, tx, batch_size=1):

            stoch_gradient, error = compute_stoch_gradient(y_, tx_, w)
            loss = MSE(error)

            w = w - gamma * stoch_gradient

            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w

def least_squares(y, tx):
    """calculate the least squares solution.
        Least squares regression using normal equations
        Should return : (w,loss)
        [last w vector + corresponding loss]
    """

    optimal_w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_loss_MSE(y, tx, optimal_w)

    return optimal_w, mse



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
