# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from implementations import *
from proj1_helpers import *

DATA_TRAIN_PATH = '../data/train.csv'
y_pre, tX_pre, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Preprocessing
tX_post = remove_aberrant_values(tX_pre)
tX_post = remove_aberrant_features(tX_post)
tX_post = rescale_outliers(tX_post)
tX_post, _ , _ = standardize(tX_post)

# Split according to categories, feature PRI_jet_num
jet = categorizing(tX_pre)

# Preprocessing
tX_test_post = remove_aberrant_values(tX_test)
tX_test_post = remove_aberrant_features(tX_test_post)
tX_test_post = rescale_outliers(tX_test_post)
tX_test_post, _ , _ = standardize(tX_test_post)

# Split according to categories, feature PRI_jet_num
jet_test = categorizing(tX_test)

# Best parameters per category :
best_degrees = [7, 9, 9, 9]
best_lambdas = [1e-2, 1e-3, 1e-3, 1e-3]

ids_pred = []
predictions = []
for i in range(len(jet)):

    degree = best_degrees[i]
    lambda_ = best_lambdas[i]

    parameters_ridge = {
      "model": "ridge",
      "param": lambda_,
    }

    tX_jet = clear_jet(tX_post[jet[i]])
    y_jet = y_pre[jet[i]]

    x_tr_p = build_poly(tX_jet, degree)

    w, _ = ridge_regression(y_jet, x_tr_p, lambda_)


    tX = clear_jet(tX_test_post[jet_test[i]])
    tX = build_poly(tX, degree)
    ids_pred.append(ids_test[jet_test[i]])
    predictions.append(predict_labels(w, tX))

ids_pred = np.concatenate(ids_pred, 0)
predictions = np.concatenate(predictions, 0)



OUTPUT_PATH = '../data/sample-submission.csv'


create_csv_submission(ids_pred, predictions, OUTPUT_PATH)
