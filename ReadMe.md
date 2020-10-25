# ReadMe
# Machine Learning Project 1 - The Higgs Boson
### Inès de Riedmatten, Yves Grandjean, Daniel Rother


##  The project
The goal of this project was to rediscover the Higgs boson starting from the actual CERN particle accelerator data using machine learning models. 

##  Code files

### Implementations.py
Contains all the functions used in Run.py, including the six basis method implementations:
+ least_squares_GD: Linear regression using gradient descent, step-size gamma. 
+ least_squares_SGD: Linear regression using stochastic gradient descent, step-size gamma.
+ least_squares: Least squares regression using normal equations.
+ ridge_regression: Ridge regression using normal equations.
+ logistic_regression: Logistic regression using gradient descent or stochastic gradient descent.
+ reg_logistic_regression: Regularized logistic regression using gradient descent or stochastic gradient descent.


### Run.py

The backbone of the project is contained in the run.py file. All the results contained in [sample_submission.csv]() can be found using this script. It contains data loading, preprocessing,  model fitting for ridge regression method, parameters and hyperparameters optimization steps. Preprocessing consists in removing the _PRI-jet-num_ categorical feature, setting the undefined variables -999.0 at the mean of the feature and standardizing the data. Feature augmentation and hyperparameter optimization were performed through grid search coupled with a 5-fold cross-validation. For least squares and ridge regression who gave the best preliminary accuracies, Data was split according to _PRI-jet-num_ feature, leading to four different sets. Our best results were found using ridge regression method on splitted data for degrees of respectively [7, 9, 9, 9] and lambdas of [1e-2, 1e-3, 1e-3, 1e-3]. Best accuracies were of [0.842, 0.807, 0.834, 0.836].


### Data folder

+ **Train and Test.csv:** Both data sets were used to train and test each different model, before evaluationg its accuracy performance. These need to be opened in order for the run.py file to compute the results. 
+ **Sample_submission.csv:** Contains the results obtained by running Run.py on our sets of data, train and test.csv


## Run the code
Here we explain to you how to export our results to the AIcrowd competition.
+ Download and extract the .zip folder of our project
+ Go to the script folder, open it in the terminal
+ Do python3 run.py file
+ The current results can be found in the sample_submission.csv file
+ Upload it on the [AIcrowd EPFL Machine Learning Higgs challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) page.
