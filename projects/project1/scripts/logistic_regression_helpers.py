import numpy as np

def logistic_loss(y, tx, w):
    """Compute logistic regression loss."""
    loss = np.sum(np.log(1 + np.exp(tx @ w))) - y @ (tx @ w)
    return loss
	
def compute_gradient_logistic_loss(y, tx, w):
    """Compute the gradient of the logistic regression."""
    return np.transpose(tx) @ (1/(1 + np.exp(- (tx @ w))) - y)
	
def regularized_logistic_regression_loss(y, tx, w, lambda_):
    """Compute the regularized logistic regression loss."""
    loss = logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T @ w)
    return loss
	
def compute_gradient_logistic_loss_regularized(y, tx, w, lambda_):
    """Compute the gradient of the regularized logistic regression """
    grad = compute_gradient_logistic_loss(y, tx, w) + 2 * lambda_ * w
    return grad