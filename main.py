# Author: Anvi Alex
# Date : 16/10/2023
# Description: A simple linear regression model built from scratch with numpy and matplotlib
#-----------

import numpy as np
from module import *

# Generate data and line to predict
np.random.seed(42)
W = 0.7
b = 0.3
start = 0
end = 1 
step = 0.02
X = np.arange(start, end, step)
Y = W*X + b

# Split the dataset
X_train, Y_train, X_test, Y_test = dataSplitter(X, Y, 0.8)

# Initialize the hyperparameters and parameters before the training phase
iteration = 100
alpha = 0.025
W = 0.5
b = 0

# Training phase
for i in range(iteration+1):
    y_hat = W * X_train  + b
    # Multiply the cost by 0.5 to somehow avoid overfitting
    cost = 0.5 * np.mean(y_hat - Y_train)**2
    # print(y_hat ==Y_train)
    dJ_dW = 2/len(X_train) * np.dot((y_hat - Y_train), X_train)
    dJ_db = 2/len(X_train) * (y_hat - Y_train)

    if i % 10 == 0 or i == 100:
        print(f'Cost: {cost} at iteration {i}')

    W = W - alpha * dJ_dW
    b = b - alpha * dJ_db

# Testing phase
y_hat_test = W * Y_test + np.mean(b)

# Plotting
plot_predictions(X_train, Y_train, X_test, Y_test,y_hat_test)


