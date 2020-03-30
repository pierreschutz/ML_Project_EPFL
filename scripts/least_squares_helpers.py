import numpy as np

def compute_gradient_MSE_loss(y, tx, w):
    """Compute the gradient."""
    N = len(tx)
    return - 1/float(N) * np.transpose(tx) @ (y - tx @ w)
	
def MSE_loss(y, tx, w):
    """Compute the mean square error."""
    e = y - tx @ w
    return 1/(2*float(len(y))) * e @ np.transpose(e)