import numpy as np
from proj1_helpers import *
from least_squares_helpers import *
from logistic_regression_helpers import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Least squares using gradient descent."""
    w = initial_w
	
    for n_iter in range(max_iters):
		# Computing the gradient of the MSE loss with respect to vector w
        gradient = compute_gradient_MSE_loss(y, tx, w)
		# Computing MSE loss
        loss = MSE_loss(y, tx, w)
		# Updating the vector w
        w -= gamma * gradient
		
    return w, loss
	

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Least square using stochastic gradient descent."""
    w = initial_w
	
    for n_iter in range(max_iters):
		# Each iteration corresponds to one epoch (num_batches=len(tx)) and each batch has size 1
        for batch_y, batch_x in batch_iter(y, tx, batch_size=1, num_batches=len(tx)):
			# Computing the gradient of the MSE loss with respect to vector w using batch_y and batch_x
            stochastic_gradient = compute_gradient_MSE_loss(batch_y, batch_x, w)
			# Updating the vector w
            w -= gamma * stochastic_gradient
        
		# Computing the MSE loss
        loss = MSE_loss(y, tx, w)

    return w, loss
	
def least_squares(y, tx):
    """Least square using normal equations."""
	
	# Computing optimal w using least squares
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
	# Computing the MSE loss
    loss = MSE_loss(y, tx, w)
	
    return w, loss
	
def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""

	# Computing w using ridge regression
    w = np.linalg.solve(tx.T @ tx + lambda_*2*len(y)*np.identity(tx.shape[1]), tx.T @ y)
	# Computing the MSE loss
    loss = MSE_loss(y, tx, w)
	
    return w, loss
	
# We decided to use the same formulas as the one introduced in class.
# Thus we needed to convert ou y labels:
# Instead of using -1 and 1 we used: 0 and 1
# Thus for this method y corresponds to the labels that are either 0 or 1.
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using stochastic gradient descent."""

    w = initial_w
    prev_loss = float('inf')
	
    for n_iter in range(max_iters):
		# Each iteration corresponds to one epoch (num_batches=len(y)) and each batch has size 1
        for batch_y, batch_x in batch_iter(y, tx, 1, num_batches=len(y)):
			# Computing the gradient of the logistic loss with respect to w
            gradient = compute_gradient_logistic_loss(batch_y, batch_x, w)
			# Updating w
            w -= gamma * gradient

		# If the loss obtained at the previous iteration is less than the actual loss we reduce gamma.
        loss = logistic_loss(y, tx, w)
        if prev_loss <= loss:
            gamma *= 0.1
        prev_loss = loss

    return w, loss

# We decided to use the same formulas as the one introduced in class.
# Thus we needed to convert ou y labels:
# Instead of using -1 and 1 we used: 0 and 1
# Thus for this method y corresponds to the labels that are either 0 or 1.
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """Regulirized logistic regression using stochastic gradient descent."""
    w = initial_w
    prev_loss = float('inf')
    
    for n_iter in range(max_iters):
		# Each iteration corresponds to one epoch (num_batches=len(y)) and each batch has size 1
        for batch_y, batch_x in batch_iter(y, tx, 1, num_batches=len(y)):
			# Computing the gradient of the logistic loss with respect to w
            gradient = compute_gradient_logistic_loss_regularized(batch_y, batch_x, w, lambda_)
			# Updating w
            w -= gamma * gradient
        
		# If the loss obtained at the previous iteration is less than the actual loss we reduce gamma.
        loss = regularized_logistic_regression_loss(y, tx, w, lambda_)
        if prev_loss <= loss:
            gamma *= 0.1
        prev_loss = loss

    return w, loss
