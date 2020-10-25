# ReadMe
# Machine Learning Project 1 - The Higgs Boson
### Inès de Riedmatten, Yves Grandjean, Daniel Rother


##  The project
The goal of this project was to rediscover the Higgs boson starting from the actual CERN particle accelerator data using machine learning models.

##  Script folder

### Implementations.py
Contains all the functions used in the project and the six basis model implementations:
+ least_squares_GD: Linear regression using gradient descent, step-size gamma.
+ least_squares_SGD: Linear regression using stochastic gradient descent, step-size gamma. Note that during each iteration, the number of batch returned is 100 and the size of each batch is 1. This was done for optimality reasons. The max_iters should be a multiple of 100.
+ least_squares: Least squares regression using normal equations.
+ ridge_regression: Ridge regression using normal equations. Lambda is the trade-off parameter of the L2-regularization term.
+ logistic_regression: Logistic regression using gradient descent, step-size gamma. The labels should take the values -1 or 1.
+ reg_logistic_regression: Regularized logistic regression using gradient descent, step size gamma. Lambda is a trade-off parameter of the penalty term. The labels should take the values -1 or 1.

All the methods return the model (the weights) and its corresponding loss. For the first four models, the loss corresponds to the mean-squared error. For the two logistic regressions, it is a negative log likelihood.  

### example.py

Contains an example of the optimization process for the Ridge regression, without splitting according to jet category. Similar codes were run for all the models in the jupyter notebook project1.ipynb. The optimization process is described in the Models optimization paragraph.


### run.py

The run.py file contains all the steps to go from training the model on train.csv to predicting the labels of the test.csv data. After loading the data from train.csv and test.csv, pre-processing consisted in removing the _PRI-jet-num_ categorical feature, removing the features with zero variance, setting the undefined variables -999.0 at the mean of the feature, replacing the data outside the three sigma rule by the values of the interval limits for each feature and standardizing the data. Note that for the test data, the rescaling of the outliers was performed with the limits found in the train data and the standardisation was also done with the mean and standard deviation of the train data. During optimization, we obtained a model of Ridge regression with degrees = [7, 9, 9, 9] and lambdas = [1e-2, 1e-3, 1e-3, 1e-3] for each jet category. The accuracies obtained were [0.842, 0.807, 0.834, 0.836] for each jet category and 0.828 overall. Finally, our predictions from the test data was tested with categorical accuracy of 0.829 and F1-score of 0.737 on AIcrowd.

### Models optimization
All models were trained and tested with the training set splitted in 80-20 proportions. Hyperoptimization and feature augmentation were performed by grid search coupled with a 5-fold cross-validation. Once models were all compared and the best (Ridge regression) was chosen, it was optimized on the splitted data according to the categorical feature _PRI-jet-num_.
The details of the optimization of each models are described in the jupyter notebook project1.ipynb.

## Data folder

+ **Train and Test.csv:** Labels of the sets are automatically attributed by the load_csv_data function. Background label (= noise) is given a value of -1 and Signal label a value of 1. Train.csv was used to find the best models. It was splitted in a training and a testing set with 80-20 proportions. Once the best model was found thanks to the grid search and the cross-validation explained above, its final accuracy was calculated by training the model on the whole train.csv and testing it on the whole test.csv set.
+ **Sample_submission.csv:** Contains the results obtained by running Run.py on our sets of data, train and test.csv


## Run the code
Here we explain to you how to export our results to the AIcrowd competition.
+ Download and extract the .zip folder of the project
+ Go to the script folder, open it in the terminal
+ $ python3 run.py
+ The current results can be found in the sample_submission.csv file
+ Upload it on the [AIcrowd EPFL Machine Learning Higgs challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) page.
