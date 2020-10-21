# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import matplotlib.pyplot as plt
import numpy as np
from proj1_helpers import predict_labels

def categorizing(x):
    categories = {0: x[:, 22] == 0,1: x[:, 22] == 1,2: x[:, 22] == 2,3: x[:, 22] == 3}
    return categories

def remove_aberrant_features(x) :

    tX = np.copy(x)
    # remove the categorical feature : Pri_jet_num
    tX = np.delete(tX, 22, 1)
    # Remove the data entries with only 0 values for all features
    #tx = tX[~np.all(tX == 0, axis = 1)]
    # Remove the features with only -999.0 values for all entries
    tX = tX[:,~np.all(tX == -999.0, axis = 0)]
    # Remove the features with only 0 values for all entries
    return tX[:,~np.all(tX == 0, axis = 0)]

def remove_aberrant_values(x):
    '''The undefined variables are set to -999.0.
    For each feature, this function will replace those values
    by the mean of the feature'''

    tX = np.copy(x)
    nb_features = tX.shape[1]
    means = np.empty(nb_features)

    # Go through each feature of x (D dimensions)
    for i in range(nb_features):
        # Calculate the mean without the abberant values
        feature_mean = tX[ tX[:,i] != -999.0, i].mean()
        nb_outliers = np.sum(tX[:,i] == -999.0)
        # Replace the abberant values by the mean of the feature
        tX[ tX[:,i] == -999.0, i] = feature_mean * np.ones(nb_outliers)
    return tX

def remove_outliers(x):
    
    tX = np.copy(x)
    nb_features = tX.shape[1]
    nb_rows = tX.shape[0]
    means = np.mean(x, axis=0)
    variances = np.std(x, axis=0)
    
    # Go through each feature of x (D dimensions)
    for i in range(nb_features):
        # Calculate the mean without the outliers
        outliers = tX[np.logical_or(tX[:,i] <= (means[i] - 3*variances[i]),tX[:,i] >= (means[i] + 3*variances[i])),i]
        sum_outliers = np.sum(np.logical_or(tX[:,i] <= (means[i] - 3*variances[i]),tX[:,i] >= (means[i] + 3*variances[i])))
        # Replace the outliers by the mean of the feature
        outliers = means[i] * np.ones(sum_outliers)
    return tX

def remove_correlation(x, par):
 
    tX = np.copy(x)
    nb_features = tX.shape[1]
    nb_rows = tX.shape[0]
    correlation = np.corrcoef(x, y=None, rowvar=False)
    
    # Go through each feature of x (D dimensions)
    for i in range(nb_features):
        # Find the correlating features
        for j in range(i):
            if correlation[i,j] >= par:
            # Delete the highly correlating feature
                tX = np.delete(tX,j,1)
    return tX

def standardize(x):
    """ Standardize the original data set = removing the mean
        & divide by the standard deviation """
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def split_data(x, y, ratio, seed=1):
    """ Split the dataset based on the split ratio.
        If ratio is 0.8 :
        80% of the data set -> training
        20% -> testing
    """

    if x.shape[0] != y.shape[0]:
        raise Exception('The lengths of x and y are not the same')
    else :
        length = x.shape[0]

    # Set seed
    np.random.seed(seed)
    # Number of training indices
    sep = int(np.floor(length * ratio))
    # Shuffle the indices
    shuffled_idx = np.random.permutation(length)
    tr_idx = shuffled_idx[:sep]
    te_idx = shuffled_idx[sep:]

    x_tr, x_te, y_tr, y_te = x[tr_idx], x[te_idx], y[tr_idx], y[te_idx]

    return x_tr, x_te, y_tr, y_te

def compute_accuracy(x, y, w):
    """ Calculate the accuracy of the prediction
        Xw compared to the known labels y
        return % accuracy """

    predictions = predict_labels(x.T, w)
    # Counts all the correct predictions
    hits = np.sum(predictions == y) / len(y) * 100

    return hits

def compute_gradient(y, tx, w):
    """Compute the gradient."""

    # Compute the error vector
    # For MSE, ▽L = -1/N*X_T*e
    error = y - np.dot(tx,w)

    return -tx.T @ error/len(y)

def compute_mse(y, tx, w):
    """ Calculate the mean squared error (MSE) loss.
    """
    error = y - tx @ w

    return 1/2 * np.mean(error**2)

def compute_mae(y, tx, w):
    """ Calculate the mean absolute error (MAE) loss.
    """
    error = y - tx @ w

    return np.mean(np.abs(error))

def sigmoid(t):
    """ Apply the sigmoid function on t."""

    # same as 1 / (1 + np.exp(-t))
    return np.exp(t)/(1 + np.exp(t))

def calculate_loss_LR(y, tx, w):
    """ Compute the loss as the negative log likelihood
        in the case of the logistic regression (LR) """

    sigma = sigmoid(tx @ w)
    loss = y.T @ np.log(sigma) + (1 - y).T @ np.log(1 - sigma)

    return -loss

def calculate_gradient_LR(y, tx, w):
    """ Compute the gradient of loss
        in the case of the logistic regression (LR)"""

    return tx.T @ (sigmoid(tx @ w) - y)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    """ Linear regression using gradient descent (GD)
        Returns : (last model w, its corresponding loss)
    """

    # Threshold for the stopping criterion
    threshold = 1e-8

    w = initial_w
    last_loss = None

    for n_iter in range(max_iters):

        # compute gradient
        gradient = compute_gradient(y, tx, w)

        # update w by gradient
        w = w - gamma * gradient

        loss = compute_mse(y, tx, w)

        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

        # Stopping criterion
        if last_loss is not None and np.abs(loss - last_loss) < threshold:
            break

        last_loss = loss

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

def least_squares_SGD(y, tx, initial_w, max_iters, batch_size, gamma):
    """ Linear regression using stochastic gradient descent (SGD)
        Returns : (last w vector, its corresponding loss)
    """

    w = initial_w

    for n_iter in range(max_iters):
        for y_,tx_ in batch_iter(y, tx, batch_size=1):

            stoch_gradient = compute_stoch_gradient(y_, tx_, w)
            loss = compute_mse(y, tx, w)

            w = w - gamma * stoch_gradient

            # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, compute_mse(y, tx, w)

def least_squares(y, tx):
    """ Calculate the least squares solution using normal equations
        Returns : (last w vector, its corresponding loss)
    """

    # Closed-form solution of the normal equation
    optimal_w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_mse(y, tx, optimal_w)

    return optimal_w, mse

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations
        lambda_ : tunes the importance of the regularizer term
        Returns : (last w vector, its corresponding loss)
    """

    # ridge regression
    a = tx.T @ tx + lambda_ * 2 * len(y) * np.identity(tx.shape[1])
    b = tx.T @ y
    optimal_w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, optimal_w)

    return optimal_w, mse

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent
        Should return : (last w vector, its corresponding loss)
    """

    w = initial_w
    # Threshold for stopping criterion
    threshold = 1e-8
    losses = []

    # From (-1,1) values to (0,1) values
    y = (y + 1)/2

    # start the logistic regression
    for iter in range(max_iters):
        # compute the loss:
        loss = calculate_loss_LR(y, tx, w)

        # compute the gradient:
        grad = calculate_gradient_LR(y, tx, w)

        # update w:
        w = w - gamma * grad

        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        losses.append(loss)

        # Stopping criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent
        Returns : (last w vector, its correspondingloss)
    """

    w = initial_w
    # Threshold for the stopping criterion
    threshold = 1e-8
    losses = []

    # From (-1,1) values to (0,1) values
    y = (y + 1)/2

    # start the logistic regression
    for iter in range(max_iters):

        loss = calculate_loss_LR(y, tx, w)
        grad = calculate_gradient_LR(y, tx, w)

        loss = loss + lambda_ / 2 * w.T @ w
        # 2nd order Taylor approx.
        grad = grad + lambda_ * w

        # update w:
        w = w - gamma * grad

        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        losses.append(loss)

        # Stopping criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss

def build_poly(x, degree):
    """ Polynomial basis functions for input data x, for j=0 up to j=degree.
        This function should return the matrix formed
        by applying the polynomial basis to the input data
        Returns x_augmented = [x**0 x**1 x**2 ... x**degree]
    """

    degree_int = int(degree)
    if degree != degree_int:
        raise Exception("Degree must be integer")

    # If the data has only 1 feature => features = 1
    features = 1 if len(x.shape) == 1 else x.shape[1]
    x = x.reshape(((x.shape[0], features)))

    # Pre-assign the x_augmented to optimize time
    x_augmented = np.ones((x.shape[0], features * degree_int + 1))
    for d in range(degree_int):

        start = d * features + 1
        stop = (d + 1) * features + 1
        x_augmented[ : , start : stop] = x**(d + 1)

    return x_augmented

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold."""
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
        max_iters = param["param"][1]
        gamma = param["param"][0]
        optimal_w, mse = least_squares_GD(y_tr, x_tr_p, initial_w, max_iters, gamma)

    elif param["model"] == "SGD":
        initial_w = param["initial_w"]
        max_iters = param["param"][1]
        batch_size = param["param"][2]
        gamma = param["param"][0]
        optimal_w, mse = least_squares_SGD(y_tr, x_tr_p, initial_w, max_iters, batch_size, gamma)

    elif param["model"] == "LS":
        optimal_w, mse =least_squares(y_tr, x_tr_p)

    elif param["model"] == "ridge":
        lambda_ = param["param"]
        # ridge regression:
        optimal_w, mse = ridge_regression(y_tr, x_tr_p, lambda_)

    elif param["model"] == "LR":
        initial_w = param["initial_w"]
        max_iters = param["param"][1]
        gamma = param["param"][0]
        # logistic regression :
        optimal_w, mse = logistic_regression(y, x_tr_p, initial_w, max_iters, gamma)

    accuracy = compute_accuracy(x_te_p, y_te, optimal_w)
    # calculate the loss for train and test data:
    loss_tr = np.sqrt(2 * mse)
    loss_te = np.sqrt(2 * compute_mse(y_te, x_te_p, optimal_w))

    return loss_tr, loss_te, optimal_w, accuracy

#def remove_undefined_variable(x):
    #'''The undefined variables are set to -999.0.
    #For each feature, this function will replace those values
    #by the mean of the feature'''

    #tX = np.copy(x)
    #nb_features = tX.shape[1]
    #means = np.empty(nb_features)

    # Go through each feature of x (D dimensions)
    #for i in range(nb_features):
        # Calculate the mean without the outliers
        #feature_mean = tX[ tX[:,i] != -999.0, i].mean()
        #nb_outliers = np.sum(tX[:,i] == -999.0)
        # Replace the outliers by the mean of the feature
        #tX[ tX[:,i] == -999.0, i] = feature_mean * np.ones(nb_outliers)

    #return tX

#def remove_aberrant_features(x) :

    #tX = np.copy(x)
    # remove the categorical feature : Pri_jet_num
    #tX = np.delete(tX, 22, 1)
    # Remove the data entries with only 0 value for all features
    #tX = tX[~np.all(tX == 0, axis = 1)]
    # Remove the features with only 0 values for all entries
    #return tX[:,~np.all(tX == 0, axis = 0)]
