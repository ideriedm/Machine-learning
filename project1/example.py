# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '../data/train.csv'
y_pre, tX_pre, ids = load_csv_data(DATA_TRAIN_PATH)

tX_post = remove_categorical_feature(tX_pre)
tX_post = remove_aberrant_values(tX_post)
tX_post, _, _ = rescale_outliers(tX_post)
# tX_post = remove_correlation(tX_post, 0.95) # Not enough useful
tX_post, mean_x , std_x = standardize(tX_post)
y_post = y_pre.copy()

##########################
# Grid search + SV, Ridge
##########################

# Choose the degree and lambdas to test
degrees = np.linspace(6, 9, 4)
lambdas = np.logspace(-5, -2, 3)
k_fold = 5

parameters_ridge = {
  "model": "ridge",
  "param": lambdas,
}

rmses_ridge, accuracy_ridge = best_degree_selection(tX_post, y_post, degrees, k_fold, parameters_ridge, seed = 1)

###################################
# Train on 80%, test on 20%, Ridge
###################################

# The best degree and lambda found above are tested
degree = 7
lambda_ = 0.01


param = { "model": "ridge" }
# Split the training data into training & testing set, ration 0.8-0.2
x_tr, x_te, y_tr, y_te = split_data(tX_post, y_post, 0.8, seed=1)

# Polynomial feature expansion
x_tr_p = build_poly(x_tr, degree)
x_te_p = build_poly(x_te, degree)

w, _ = ridge_regression(y_tr, x_tr_p, lambda_)
accuracy = compute_accuracy(x_te_p, y_te, w, param)
rmse = np.sqrt(2 * compute_mse(y_te, x_te_p, w))

print("Test model : ", param["model"], "degree = ", degree, "accuracy = ", accuracy, "RMSE =", rmse)
