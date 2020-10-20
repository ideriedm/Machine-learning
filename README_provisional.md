# ReadMe
# Machine Learning Project 1 - The Higgs Boson
### Inès de Riedmatten, Yves Grandjean, Daniel Rother


##  The project
The goal of this project was to rediscover the Higgs boson starting from the actual CERN particle accelerator data using machine learning models. 

##  Code files
### Run.py

The backbone of the project is contained in the run.py file. All the results contained in [sample_submission.csv]() can be found using this script. It contains data loading, preprocessing, model fitting for several methods, parameters and hyperparameters optimization steps. Additionaly, the plots of some key graphs that we used to get a better understanding of the problem can also be found in it. Our best results were found using (**ADD THE RESULTS**)

### Implementations.py
Contains all the functions that we use in Run.py, including the 6 basis method implementations such as least_squares_GD, least_squares_SGD, etc.

### Data folder

+ **Train and Test.csv:** Both data sets were used to train and test each different model, before evaluationg its accuracy performance
+ **Sample_submission.csv:** Contains the results obtained by running Run.py on our sets of data, train and test.csv

## Run the code
Here we explain to you how to export our results to the AIcrowd competition.
+ Download and extract the .zip folder of our project
+ **HOW DO WE WANT TO MAKE THEM RUN IT**
+ The current results can be found in the sample_submission.csv file
+ Upload it on the [AIcrowd EPFL Machine LEarning Higgs challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) page.