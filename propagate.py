from sigmoid import sigmoid
import numpy as np
def propagate(w, b, X, Y):
    m = X.shape[1]

    # FORWARD PROPAGATION
    # compute activation (A)
    A = sigmoid(np.dot(w.T, X) + b)

    # compute cost (logistic regression)
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    
    # BACKWARD PROPAGATION
    dw = 1/m*np.dot(X, (A-Y).T)
    db = 1/m*(np.sum(A-Y))

    # make dictionary of gradients
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
