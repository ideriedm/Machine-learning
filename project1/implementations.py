# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import matplotlib.pyplot as plt
import numpy as np
from proj1_helpers import predict_labels


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    if x.shape[0] != y.shape[0]:
        raise Exception('The length of x and y are not the same')
    else :
        length = x.shape[0]

    # set seed
    np.random.seed(seed)
    # number of training indices
    sep = int(np.floor(length*ratio))
    # shuffle the indices
    shuffled_idx = np.random.permutation(length)
    tr_idx = shuffled_idx[:sep]
    te_idx = shuffled_idx[sep:]

    x_tr, x_te, y_tr, y_te = x[tr_idx], x[te_idx], y[tr_idx], y[te_idx]

    return x_tr, x_te, y_tr, y_te

def compute_accuracy(x, y, w):
    """ Calculate the accuracy of the prediction
    Xw compared to the labels y
    return % accuracy """
    predictions = predict_labels(x.T, w)
    hits = np.sum(predictions == y)/len(y)*100

    return hits


def compute_gradient(y, tx, w):
    """Compute the gradient."""

    # compute gradient and error vector
    # For MSE, ▽L = -1/N*X_T*e
    error = y - np.dot(tx,w)

    # tx.T = tx.transpose()
    return -np.dot(tx.T,error)/len(y)

def compute_mse(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    error = y - np.dot(tx,w)
    # or y - tx @ w
    # or y - tx.dot(w)
    return 1/2*np.mean(error**2)

def compute_mae(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    error = y - np.dot(tx,w)
    # or y - tx @ w
    # or y - tx.dot(w)
    return np.mean(np.abs(error))

def grid_search(y, tx, w0, w1):
     """Algorithm for grid search."""
     losses = np.zeros((len(w0), len(w1)))
    #compute loss for each combination of w0 and w1.
     for i in range(w0.shape[0]):
         for j in range(w1.shape[0]):
             w = np.array([w0[i],w1[j]])
             losses[i,j] = compute_loss(y,tx,w)

     return losses


def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    """Linear regression using gradient descent
        Should return : (w,loss)
        [last w vector + corresponding loss]
    """

    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        # or loss = MAE(error)

        # update w by gradient
        w = w - gamma * gradient

        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss

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

            stoch_gradient = compute_stoch_gradient(y_, tx_, w)
            loss = compute_mse(y, tx, w)

            w = w - gamma * stoch_gradient

            # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution.
        Least squares regression using normal equations
        Should return : (w,loss)
        [last w vector + corresponding loss]
    """

    optimal_w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_mse(y, tx, optimal_w)

    return optimal_w, mse



def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations
        Should return : (w,loss)
        [last w vector + corresponding loss]
    """

    # ridge regression:
    a = tx.T @ tx + lambda_ * 2 * len(y) * np.identity(tx.shape[1])
    b = tx.T @ y
    optimal_w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, optimal_w)

    return optimal_w, mse

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

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data

    # if the data has only 1 feature => features = 1
    features = 1 if len(x.shape) == 1 else x.shape[1]
    x = x.reshape(((x.shape[0], features)))
    x_augmented = np.ones((x.shape[0], int(features*degree + 1)))
    for d in range(degree):
        start = int(d*features+1)
        stop = int((d+1)*features+1)
        x_augmented[:,start : stop] = x**(d+1)
    return x_augmented

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, param, degree):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train:
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    x_tr = np.delete(x, k_indices[k], 0)
    y_tr = np.delete(y, k_indices[k], 0)


    # form data with polynomial degree:
    x_tr_p = build_poly(x_tr, degree)
    x_te_p = build_poly(x_te, degree)

    if param["model"] == "GD":
        initial_w = param["initial_w"]
        max_iters = param["param"]
        gamma = 0.3
        optimal_w, mse = least_squares_GD(y_tr, x_tr_p, initial_w, max_iters, gamma)

    elif param["model"] == "SGD":
        initial_w = param["initial_w"]
        max_iters = param["param"]
        gamma = 0.3
        optimal_w, mse = least_squares_SGD(y_tr, x_tr_p, initial_w, max_iters, gamma)

    elif param["model"] == "LS":
        optimal_w, mse =least_squares(y_tr, x_tr_p)

    elif param["model"] == "ridge":
        lambda_ = param["param"]
        # ridge regression:
        optimal_w, mse = ridge_regression(y_tr, x_tr_p, lambda_)

    accuracy = compute_accuracy(x_te_p, y_te, optimal_w)
    # calculate the loss for train and test data:
    loss_tr = np.sqrt(2 * mse)
    loss_te = np.sqrt(2 * compute_mse(y_te, x_te_p, optimal_w))

    return loss_tr, loss_te, optimal_w, accuracy

def remove_undefined_variable(x):
    '''The undefined variables are set to -999.0.
    For each feature, this function will replace those values
    by the mean of the feature'''

    tX = np.copy(x)
    nb_features = tX.shape[1]
    means = np.empty(nb_features)

    # Go through each feature of x (D dimensions)
    for i in range(nb_features):
        # Calculate the mean without the outliers
        feature_mean = tX[ tX[:,i] != -999.0, i].mean()
        nb_outliers = np.sum(tX[:,i] == -999.0)
        # Replace the outliers by the mean of the feature
        tX[ tX[:,i] == -999.0, i] = feature_mean * np.ones(nb_outliers)

    return tX

def remove_aberrant_features(x) :

    tX = np.copy(x)
    # remove the categorical feature : Pri_jet_num
    tX = np.delete(tX, 22, 1)
    # Remove the data entries with only 0 value for all features
    tX = tX[~np.all(tX == 0, axis = 1)]
    # Remove the features with only 0 values for all entries
    return tX[:,~np.all(tX == 0, axis = 0)]
