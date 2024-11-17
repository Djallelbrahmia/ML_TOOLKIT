from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np 
from Models.none_linear_models.svm import SVM
from Models.utils.svm_kernels import linear_kernel ,rbf_kernel,polynomial_kernel
from CostFunctions import MeanSquaredError


iris = load_iris()
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [5, 5],
    [1, 0],
    [0, 1],
    [3, 1],
    [4, 4]
])

# Corresponding labels (y)
y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# Split the data into training and testing sets

# Create SVM with different kernels
# Linear kernel
svm_linear = SVM(kernel_func=polynomial_kernel(degree=2, gamma=0.1, coef0=1.0))

# RBF kernel with gamma=0.1
svm_rbf = SVM(kernel_func=rbf_kernel(gamma=0.1))

# Polynomial kernel (quadratic) with custom parameters
svm_poly = SVM(kernel_func=polynomial_kernel(degree=2, gamma=0.1, coef0=1.0))


svm_linear.fit(X, y)

# Make predictions
predictions = svm_linear.predict(X)
mse=MeanSquaredError()
print("Mse Error using linear kernal : " ,mse.compute(predictions=predictions,y=y) )
