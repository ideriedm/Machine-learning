
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2

from proj1_helpers import * 
DATA_TRAIN_PATH = '/home/ML_course/projects/project1/data/train.csv' # CHECK THAT YOUR DATA IS HERE -> add this to README file 
data_y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#data standardization
centered_data = tX - np.mean(tX, axis=0)
std_data = centered_data / np.std(centered_data, axis=0)

#covariance matrix
covariance_matrix = np.cov(tX.T) 
#covariance_matric is 30x30
#eigenvalues and vectors decomposition
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print("Eigenvector: \n",eigen_vectors,"\n")
print("Eigenvalues: \n", eigen_values, "\n")

#PC1 is the first column of eigen_vectors, PC2 is the second,...

#how much of our data is explained by each one of these components?
variance_explained = []
for i in eigen_values:
     variance_explained.append((i/sum(eigen_values))*100)
        
print(variance_explained)
#here we have 74.33% of the variance of our data is explained by PC1

#identifying components for 95% of variance explained
cumulative_variance_explained = np.cumsum(variance_explained)
print(cumulative_variance_explained)

#visualize the variance explained and finding the "elbow"
x = np.arange(1,tX.shape[1]+1) 
y = cumulative_variance_explained  
plt.title("Variance explained by the principal components") 
plt.xlabel("Principal components") 
plt.ylabel("Percentage of the variance explained") 
plt.plot(x,y) 
plt.show()

#only need to use the 3 first PCs as they account for 97.7% of the variance
projection_matrix = (eigen_vectors.T[:][:3]).T
print(projection_matrix)

tX_pca = tX.dot(projection_matrix)
print(tX_pca)

