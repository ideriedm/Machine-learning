# ReadMe
# Machine Learning Project 1 - The Higgs Boson
### Inès de Riedmatten, Yves Grandjean, Daniel Rother


##  The project
The goal of this project was to rediscover the Higgs boson starting from the actual CERN particle accelerator data using machine learning models. 

##  Code files

### Implementations.py
Contains all the functions used in Run.py and the six basis method implementations:
+ least_squares_GD: Linear regression using gradient descent, step-size gamma. 
+ least_squares_SGD: Linear regression using stochastic gradient descent, step-size gamma.
+ least_squares: Least squares regression using normal equations.
+ ridge_regression: Ridge regression using normal equations. Lambda is the tradeoff parameter of the L2-regularization term.
+ logistic_regression: Logistic regression using gradient descent, step-size gamma. The labels should take the values -1 or 1.
+ reg_logistic_regression: Regularized logistic regression using gradient descent, step size gamma. Lambda is a tradeoff parameter of the penalty term. The labels should take the values -1 or 1. 


### Run.py

The run.py file contains all the steps to go from training the model on train.csv to predict the labels of the test.csv data. After loading the data from train.csv and test.csv. Preprocessing consisted in removing the _PRI-jet-num_ categorical feature, removing the features with zero variance, setting the undefined variables -999.0 at the mean of the feature, replacing the data outside the three sigma rule by the values of the interval limits for each feature and standardizing the data. Note that for the test data, the rescaling of the outliers was performed with the limits found in the train data and the standardisation was also done with the mean and standard deviation of the train data. During optimization, we obtained a model of Ridge regression with degrees = [7,9,9,9] and lambdas = [1e-2,1e-3,1e-3,1-3] for each jet category. The accuracies obtained were [0.842, 0.807, 0.834, 0.836] for each jet category and 0.828 overall. Finally, our predictions from the test data was tested with categorical accuracy of 0.829 and F1-score of 0.737 on AIcrowd.


### Data folder

+ **Train and Test.csv:** Labels of the sets are automatically attributed by the load_csv_data function. Background label (=noise) is given -1 value and Signal value 1. Train.csv was used to find the best models. It was splitted in a training and a testing set with 80-20 proportions. Once the best model was found, its final accuracy was calculated by training the model on the whole train.csv and tested on the whole test.csv set.
+ **Sample_submission.csv:** Contains the results obtained by running Run.py on our sets of data, train and test.csv


## Run the code
Here we explain to you how to export our results to the AIcrowd competition.
+ Download and extract the .zip folder of our project
+ Go to the script folder, open it in the terminal
+ Do python3 run.py file
+ The current results can be found in the sample_submission.csv file
+ Upload it on the [AIcrowd EPFL Machine Learning Higgs challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) page.
