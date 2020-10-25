# -*- coding: utf-8 -*-
import numpy as np
from proj1_helpers import *

##############################################
## Data - preprocessing
##############################################

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """ Polynomial basis functions for input data x, for j=0 up to j=degree.
        This function should return the matrix formed by applying the polynomial
        basis to the input data.
        By convention, degree = 0 returns x
        Arguments :
            x : dataset
            degree : the degree until which the function will augment the x
        Returns :
            x_augmented = [x**0 x**1 x**2 ... x**degree]
            Note : x**0 corresponds to the first column filled with 1
    """

    # Check that the user provide an integer
    degree_int = int(degree)
    if degree != degree_int:
        raise Exception("Degree must be integer")

    if degree != 0:
        # If the data has only 1 feature => features = 1
        features = 1 if len(x.shape) == 1 else x.shape[1]
        x = x.reshape(((x.shape[0], features)))

        # Pre-assign the x_augmented to optimize time
        x_augmented = np.empty((x.shape[0], features * degree_int + 1))
        # The first column is filled with ones to add a bias
        x_augmented[:, 0] = np.ones(x.shape[0])
        for d in range(degree_int):

            start = d * features + 1
            stop = (d + 1) * features + 1
            x_augmented[ : , start : stop] = x**(d + 1)

        return x_augmented

    elif degree == 0:
        return x

def categorizing(x):
    """ The feature 23 (PRI_JET_NUM) is a categorical variables that contains
        0, 1, 2 or 3.
        Argument :
            x : dataset
        Returns :
            E.g. categories[0] = vector of boolean of length x.shape[0] = N.
                 Contains TRUE if the x[:, 22] = 0
                 x[categories[0]] returns all the x that have 0 as PRI_JET_NUM
    """
    categories = {i: x[:, 22] == i for i in range(4)}

    return categories

def standardize(x, mean = None, std = None):
    """ Standardize the original data set
        Argument :
            x : dataset
        Returns :
            standardized data set
            mean along the columns
            standard deviation along the columns
     """
    if mean is None or std is None:
        mean_x = np.mean(x, axis = 0)
        x = x - mean_x
        std_x = np.std(x, axis = 0)
        x = x / std_x
        return x, mean_x, std_x
    else:
        return (x-mean)/std, mean, std

def split_data(x, y, ratio, seed=1):
    """ Split the dataset based on the split ratio.
        If ratio is 0.8 :
        80% of the data set -> training
        20% -> testing
        Argument :
            x : dataset
            y : known labels
            ratio : ratio of splitting
            seed : seed for the random seed
        Returns :
            x_tr, y_tr : the x and y data for Training
            x_te, y_te : the x and y data for testing
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
    # Training indices
    tr_idx = shuffled_idx[:sep]
    # Testing indices
    te_idx = shuffled_idx[sep:]

    x_tr, x_te, y_tr, y_te = x[tr_idx], x[te_idx], y[tr_idx], y[te_idx]

    return x_tr, x_te, y_tr, y_te

def remove_categorical_feature(x) :

    """ Removes the categorical feature 23 Pri_jet_num
        Argument :
            x : dataset
        Returns :
            tX : dataset without the Pri_jet_num feature
    """

    # remove the categorical feature : Pri_jet_num
    return np.delete(x, 22, axis=1)

def clear_variance_0(x):

    """ Removes the features with 0 variance (columns filled with the same value)
        Argument :
            x : dataset
        Returns :
            x : dataset without the 0-variance features
    """

    return np.delete(x, np.argwhere(np.std(x, axis = 0) < 1e-5), axis=1)

def remove_aberrant_values(x):
    ''' The undefined variables are set to -999.0.
        For each feature, this function will replace those values
        by the mean of the feature
        Argument :
            x : dataset
        Returns :
            tX : dataset with the -999.0 replaced by the mean of the feature
    '''

    # Matrix of size NxD containing true where x = -999
    aberrant_idxs = x == -999
    # Replace -999 by nan
    x_nan = np.where(aberrant_idxs, np.nan, x)
    # Mean without nan of each features
    features_mean = np.nanmean(x_nan, axis=0)
    # Matrix of size NxD. Each row is a copy of features_mean vector
    features_mean_mat = np.repeat(features_mean[np.newaxis,:], x.shape[0], axis=0)
    # Replace the aberrant idx by the corresponding mean of the feature
    return np.where(aberrant_idxs, features_mean_mat, x)

def rescale_outliers(tX, lim_min=None, lim_max=None):
    """ Replaces the outliers that are outside the interval :
        mean(x:d) +/- 3 * std(x:d) by the bound values of the interval
        Argument :
            x : dataset
        Returns :
            tX : dataset with the points outside mean +/- 3*std replaced
            at the extreme values of the interval
    """
    if lim_min is None or lim_max is None:
        # Calculate the means & the variances of all the features
        means = np.mean(tX, axis = 0)
        variances = np.std(tX, axis = 0)
        # Calculate the minimal and the maximal limit of the confidence interval
        lim_min = means - 3 * variances
        lim_max = means + 3 * variances

    # Test along each column :
    # If tX[:,d] > lim_max[d] => tX[:,d] = lim_max[d]
    # If tX[:,d] < lim_min[d] => tX[:,d] = lim_min[d]
    return np.clip(tX, lim_min, lim_max), lim_min, lim_max

def get_exceeding_cols(mat, thresh):
    """ Argument :
        mat : matrix
        thresh : the highest normalized correlation tolerated.
        Returns :
            the sorted indices where the maximum of the absolute value of each
            column of the upper diagonal matrix is greater than the threshold
    """

    # Search the max of the absolute value in the triangular upper matrix
    # k = 1 => the triangular matrix starts 1 diagonal above the main one
    max_in_cols = np.max(np.abs(np.triu(mat, k=1)), axis=0)
    # Get the sorted indices from the max column values
    # (Sorted from the biggest one to the smallest one)
    idxs_sort = np.argsort(max_in_cols)[::-1]
    # Get the indices where the max column values exceed the threshold
    idxs_exceeding = np.argwhere(max_in_cols[idxs_sort] > thresh)
    return idxs_sort[idxs_exceeding]

def remove_correlation(x, corr_lim):
    """ Removes the highly (anti-)correlated features, i.e. if the absolute value
        of the correlation between column i & j >= param, based on Pearson
        product-moment correlation coefficients
        Argument :
            x : dataset
            corr_lim : the highest normalized correlation tolerated.
        Returns :
            tX : dataset without the highly correlated features
    """
    t_x_uncorr = x

    while True:
        # Each row corresponds to a variable, so transpose to have
        # each variable = each feature
        corr = np.corrcoef(t_x_uncorr.T)
        exceed_idxs = get_exceeding_cols(corr, corr_lim)
        # Continue until no exceeding index
        if not exceed_idxs.size:
            break
        t_x_uncorr = np.delete(t_x_uncorr, exceed_idxs[0], axis=1)
    return t_x_uncorr

##############################################
## Costs
##############################################

def compute_mse(y, tx, w):
    """ Calculate the mean squared error (MSE) loss.
        Argument :
            x : dataset
            y : known labels
            w : model to be tested
        Returns :
            mse : the mean squared error
    """
    error = y - tx @ w

    return 1/2 * np.mean(error**2)

def compute_mae(y, tx, w):
    """ Calculate the mean absolute error (MAE) loss.
        Argument :
            x : dataset
            y : known labels
            w : model to be tested
        Returns :
            mae : the mean absolute error
    """
    error = y - tx @ w

    return np.mean(np.abs(error))

def calculate_loss_LR(y, tx, w):
    """ Compute the loss as the negative log likelihood
        in the case of the logistic regression (LR).
        The loss was normalized by the length of y.
        Argument :
            x : dataset
            y : known labels
            w : model to be tested
        Returns :
            mse : the mean squared error
    """

    # sigma = sigmoid(tx @ w)
    sigma = tx @ w

    loss = (np.sum(np.log(1 + np.exp(sigma))) - y.T @ sigma)/len(y)
    # loss = -(y.T @ np.log(sigma) + (1 - y).T @ np.log(1 - sigma))

    return np.squeeze(loss)

##############################################
## Gradients
##############################################

def compute_gradient(y, tx, w):
    """ Compute the gradient.
        Argument :
            x : dataset
            y : known labels
            w : model of the previous iteration
        Returns :
            gradient : the new gradient to use for the next iteration
    """

    # Compute the error vector
    # For MSE, ▽L = -1/N*X_T*e
    error = y - np.dot(tx,w)

    return -tx.T @ error/len(y)

def calculate_gradient_LR(y, tx, w):
    """ Compute the gradient of loss
        in the case of the logistic regression (LR)
        Argument :
            x : dataset
            y : known labels
            w : model of the previous iteration
        Returns :
            gradient : the new gradient to use for the next iteration
    """

    return (tx.T @ (sigmoid(tx @ w) - y))

##############################################
## Additional helpers
##############################################

def sigmoid(t):
    """ Apply the sigmoid function on t
        Argument :
            t : data
        Returns :
            sigmoid(t)
    """

    return 1.0/(1 + np.exp(-t))

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
            yield batch_num, shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

##############################################
## Models
##############################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
        Linear regression using gradient descent (GD)
        Argument :
            tx : dataset
            y : known labels
            initial_w : starting/initial model
            max_iters : the maximal number of iterations the function can do
            gamma : the learning rate / step size
        Returns :
            w : the last Model
            loss : its corresponding MSE loss
    """

    # Threshold for the stopping criterion
    threshold = 1e-8

    w = initial_w
    last_loss = None

    for n_iter in range(max_iters):

        # Compute gradient
        gradient = compute_gradient(y, tx, w)

        # update w by gradient
        w = w - gamma * gradient

        loss = compute_mse(y, tx, w)

        # Stopping criterion
        if last_loss is not None and np.abs(loss - last_loss) < threshold:
            return w, loss

        last_loss = loss

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, batch_size, gamma):
    """ Linear regression using stochastic gradient descent (SGD)
        Argument :
            tx : dataset
            y : known labels
            initial_w : starting/initial model
            max_iters : the maximal number of iterations the function can do
            batch_size : the size of the batch used in batch_iter
            gamma : the learning rate / step size
        Returns :
            w : the last Model
            loss : its corresponding MSE loss
    """

    # Threshold for the stopping criterion
    threshold = 1e-8

    w = initial_w
    last_loss = None
    # See function batch_iter
    num_batches = 100

    for n_iter in range(int(max_iters / num_batches)):
        # To increase the efficiency of the function, batch_iter is called with
        # batch_size = 1, but num_batches = 100. Thus, tx is shuffled only 1 time
        # per 100 call.
        for i, y_, tx_ in batch_iter(y, tx, batch_size = batch_size,
                                     num_batches = num_batches):
            # For 1 n_iter, the compute_gradient is called num_batches time
            stoch_gradient = compute_gradient(y_, tx_, w)

            w = w - gamma * stoch_gradient

            loss = compute_mse(y, tx, w)

            # Stopping criterion
            # if last_loss is not None and np.abs(loss - last_loss) < threshold:
            #     print("Converged in : ", n_iter * num_batches + i, " iterations")
            #     return w, loss

            last_loss = loss

    return w, loss

def least_squares(y, tx):
    """ Calculate the least squares solution using normal equations
        Argument :
            tx : dataset
            y : known labels
        Returns :
            w : the model
            loss : its corresponding MSE loss
    """

    # Closed-form solution of the normal equation
    optimal_w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_mse(y, tx, optimal_w)

    return optimal_w, mse

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations
        Argument :
            tx : dataset
            y : known labels
            lambda_ : tunes the importance of the regularizer term
        Returns :
            w : the model
            loss : its corresponding MSE loss
    """

    # ridge regression
    a = tx.T @ tx + lambda_ * 2 * len(y) * np.identity(tx.shape[1])
    b = tx.T @ y
    optimal_w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, optimal_w)

    return optimal_w, mse

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent
        Argument :
            tx : dataset
            y : known labels. Should be either -1 or 1
            initial_w : starting/initial model
            max_iters : the maximal number of iterations the function can do
            gamma : the learning rate / step size
        Returns :
            w : the last Model
            loss : its corresponding LR loss
    """

    w = initial_w
    # Threshold for stopping criterion
    threshold = 1e-8

    last_loss = None

    if not np.unique(y) in np.r_[-1,1]:
        raise Exception("The labels should be either -1 or 1.")
    # From (-1,1) values to (0,1) values
    y = (y + 1)/2

    for n_iter in range(max_iters):

        # Compute the gradient:
        grad = calculate_gradient_LR(y, tx, w)

        # Update w:
        w = w - gamma * grad

        # Compute the loss:
        loss = calculate_loss_LR(y, tx, w)

        # # log info
        # if iter % 100 == 0:
        #     print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # losses.append(loss)

        # Stopping criterion
        if last_loss is not None and np.abs(loss - last_loss) < threshold:
            print("Converged in : ", n_iter * num_batches + i, " iterations")
            return w, loss

        last_loss = loss


    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent.
        Additional penality term in the loss function
        Argument :
            tx : dataset
            y : known labels. Should be either -1 or 1
            lambda_ : tunes the penalization of the model
            initial_w : starting/initial model
            max_iters : the maximal number of iterations the function can do
            gamma : the learning rate / step size
        Returns :
            w : the last Model
            loss : its corresponding LR loss
    """

    w = initial_w
    # Threshold for the stopping criterion
    threshold = 1e-8
    last_loss = None

    if not np.unique(y) in np.r_[-1,1]:
        raise Exception("The labels should be either -1 or 1.")
    # From (-1,1) values to (0,1) values
    y = (y + 1)/2

    for n_iter in range(max_iters):

        # Compute the gradient
        grad = calculate_gradient_LR(y, tx, w)

        # 2nd order Taylor approx.
        grad = grad + 2 * lambda_ * w

        # Update w:
        w = w - gamma * grad

        # Calculate the loss of the new model
        loss = calculate_loss_LR(y, tx, w)
        loss = loss + lambda_ * np.squeeze( w.T @ w )

        # # log info
        # if iter % 100 == 0:
        #     print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # losses.append(loss)

        # Stopping criterion
        if last_loss is not None and np.abs(loss - last_loss) < threshold:
            print("Converged in : ", n_iter , " iterations")
            return w, loss

        last_loss = loss

    return w, loss


##############################################
## Model - testing
##############################################

def compute_accuracy(x, y, w, param):
    """ Calculate the accuracy of the prediction
        Xw compared to the known labels y
        Argument :
            x : dataset
            y : known labels
            w : model to test
            param : dictionnary containing the name of the model used
                    E.g. : param = { "model" : "GD" }
                           param = { "model" : "SGD" }
                           param = { "model" : "LS" }
                           param = { "model" : "ridge" }
                           param = { "model" : "LR" }
                           param = { "model" : "REG_LR" }
        Returns :
            % accuracy
    """

    if param["model"] != "LR" and param["model"] != "REG_LR":
        predictions = predict_labels(w, x)
    else :
        predictions = predict_labels_LR(w, x)

    # Counts all the correct predictions
    hits = np.sum(predictions == y) / len(y) * 100

    return hits

def cross_validation(y, x, k_indices, k, param, degree):
    """ Cross validation (CV) of the models
        Argument :
            y : known labels
            x : dataset
            k_indices : vector of indices for Training. See build_k_indices
            k : indicate in which k-fold the CV is
            param : dictionnary containing the information of the model used
                    E.g. :
                    gammas and lambdas : vectors of parameters to test
                    max_iters : the number of maximal iterations
                    batch_size : The size of the batch used for SGD
                    parameters_GD = { "model": "GD", "param": [gammas, max_iters]}
                    parameters_SGD = { "model": "SGD","param": [gammas, max_iters, batch_size]}
                    parameters_LS = { "model": "LS" }
                    parameters_ridge = { "model": "ridge", "param": lambdas}
                    parameters_LR = { "model": "LR", "param": [gammas, max_iters]}
                    parameters_REG_LR = { "model": "REG_LR", "param": [lambdas, max_iters, gamma]}
            degree : degree until which the x will be augmented. See build_poly
        Returns :
            loss_tr : loss of training set
            loss_te : loss of testing set
            optimal_w : the model retained
            accuracy : the accuracy of the model
    """

    # Get k'th subgroup in test, others in train:
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    x_tr = np.delete(x, k_indices[k], 0)
    y_tr = np.delete(y, k_indices[k], 0)

    # By convention :
    # degree = 0 => x_tr_p = x_tr (no bias added)
    # degree = 1 => x_tr_p = [1 x_tr] (the first column is filled with 1, to add a bias)
    if degree != 0:
        # form data with polynomial degree:
        x_tr_p = build_poly(x_tr, degree)
        x_te_p = build_poly(x_te, degree)
    else :
        x_tr_p = x_tr
        x_te_p = x_te

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
        optimal_w, mse = least_squares(y_tr, x_tr_p)

    elif param["model"] == "ridge":
        lambda_ = param["param"]
        optimal_w, mse = ridge_regression(y_tr, x_tr_p, lambda_)

    elif param["model"] == "LR":
        initial_w = param["initial_w"]
        max_iters = param["param"][1]
        gamma = param["param"][0]
        optimal_w, loss = logistic_regression(y_tr, x_tr_p, initial_w, max_iters, gamma)

    elif param["model"] == "REG_LR":
        initial_w = param["initial_w"]
        max_iters = param["param"][1]
        gamma = param["param"][2]
        lambda_ = param["param"][0]
        # regularized logistic regression :
        optimal_w, loss = reg_logistic_regression(y_tr, x_tr_p, lambda_, initial_w, max_iters, gamma)


    accuracy = compute_accuracy(x_te_p, y_te, optimal_w, param)

    # Calculate the loss for train and test data
    # For all the models except LR & LOG_LR, we compute the RMSE
    if param["model"] != 'LR' and param["model"] != 'REG_LR':
        loss_tr = np.sqrt(2 * mse)
        loss_te = np.sqrt(2 * compute_mse(y_te, x_te_p, optimal_w))
    else:
        # The loss corresponds to the negative log likelihood
        loss_tr = loss
        # The loss is computed with y of value [0,1], not [-1,1]
        loss_te = calculate_loss_LR((y_te + 1)/2, x_te_p, optimal_w)

    return loss_tr, loss_te, optimal_w, accuracy

def best_degree_selection(x, y, degrees, k_fold, model, seed = 1):
    """ Optimize the best degree and/or the best parameter for each model
    Argument :
        x : dataset
        y : known labels
        degrees : List of degrees to test.
                  For each degree, x will be augmented until degree.
                  See build_poly
        k_fold : The number of k_fold to perfom in the cross_validation.
                 The RMSEs & the accuracies are average over the k_fold
        model : dictionnary containing the information of the model used
                E.g. :
                gammas and lambdas : vectors of parameters to test
                max_iters : the number of maximal iterations
                batch_size : The size of the batch used for SGD

                parameters_GD = { "model": "GD", "param": [gammas, max_iters]}
                parameters_SGD = { "model": "SGD","param": [gammas, max_iters, batch_size]}
                parameters_LS = { "model": "LS" }
                parameters_ridge = { "model": "ridge", "param": lambdas}
                parameters_LR = { "model": "LR", "param": [gammas, max_iters]}
                parameters_REG_LR = { "model": "REG_LR", "param": [lambdas, max_iters, gamma]}
    Returns :
        losses_list : list of the losses per all couple (degree, param).
                      All the losses are RMSE, except for LR & REG_LR where
                      they are normalized negative log likelihood.
        accuracy_list : list of the accuracies per all couple (degree, param)
    """
    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    print("Method = ", model["model"])

    if model["model"] != 'LS':
        if model["model"] != 'ridge':
            list_param = model["param"][0] # gammas or lambdas
        else :
            list_param = model["param"]
        # Store all the RMSEs
        losses_list = np.empty([degrees.size, list_param.size])
        accuracy_list = np.empty([degrees.size, list_param.size])

    else:
        losses_list = np.empty(degrees.size)
        accuracy_list = np.empty(degrees.size)

    # Vary degree
    for i, degree in enumerate(degrees):

        print("Degree = ", degree)
        print("Progress : ", i/len(degrees)*100 , " % ")

        if model["model"] != 'LS' and model["model"] != 'ridge':
            if degree != 0:
                model["initial_w"] = np.zeros(int(x.shape[1] * degree + 1))
            else :
                model["initial_w"] = np.zeros(int(x.shape[1]))

        if model["model"] != 'LR':
            if degree != 0:
                model["initial_w"] = np.ones(int(x.shape[1] * degree + 1))/1000
            else :
                model["initial_w"] = np.zeros(int(x.shape[1]))

        ####################################
        # Not LS : optimize lambda or gamma
        ####################################

        if model["model"] != 'LS':

            for j, p in enumerate(list_param):

                loss_te_tmp = np.empty(k_fold)
                accuracy_tmp = np.empty(k_fold)

                if model["model"] != 'ridge':
                    model["param"][0] = p
                else :
                    model["param"] = p

                for k in range(k_fold):
                    _, loss_te, _, accuracy = cross_validation(y, x, k_indices, k, model, degree)
                    loss_te_tmp[k] = loss_te
                    accuracy_tmp[k] = accuracy

                losses_list[i,j] = np.mean(loss_te_tmp)
                accuracy_list[i,j] = np.mean(accuracy_tmp)

        ######################################
        # LS : No other parameter to optimize
        ######################################

        else :
            loss_te_tmp = np.empty(k_fold)
            accuracy_tmp = np.empty(k_fold)

            for k in range(k_fold):
                    _, loss_te, _, accuracy = cross_validation(y, x, k_indices, k, model, degree)
                    loss_te_tmp[k] = loss_te
                    accuracy_tmp[k] = accuracy

            losses_list[i] = np.mean(loss_te_tmp)
            accuracy_list[i] = np.mean(accuracy_tmp)

    print("The method used is : ", model["model"])

    losses_list = np.squeeze(losses_list)
    accuracy_list = np.squeeze(accuracy_list)

    # Print the results corresponding to the smallest RMSE
    if model["model"] == 'LS':
        idx_best_degree = np.squeeze(np.argwhere(losses_list == np.min(losses_list)))
        print("The best degree is", degrees[idx_best_degree],
              " with RMSE = ", losses_list[idx_best_degree],
              "and accuracy = ", accuracy_list[idx_best_degree])

    elif model["model"] != 'LR' and model["model"] != 'REG_LR':
        idx_best_degree = np.squeeze(np.argwhere(losses_list == np.min(losses_list)))
        print("The best degree is", degrees[idx_best_degree[0]],
              " best parameter is", list_param[idx_best_degree[1]],
              " with RMSE = ", losses_list[idx_best_degree[0],idx_best_degree[1]],
              "and accuracy = ", accuracy_list[idx_best_degree[0],idx_best_degree[1]])
    else:
        idx_best_degree = np.squeeze(np.argwhere(losses_list == np.min(losses_list)))
        print("The best degree is", degrees[idx_best_degree[0]],
              " best parameter is", list_param[idx_best_degree[1]],
              " with loss NLL = ", losses_list[idx_best_degree[0],idx_best_degree[1]],
              "and accuracy = ", accuracy_list[idx_best_degree[0],idx_best_degree[1]])


    return losses_list, accuracy_list
