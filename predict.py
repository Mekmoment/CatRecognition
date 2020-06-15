from sigmoid import sigmoid
import numpy as np
def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute Activation as Vector
    A = sigmoid(np.dot(w.T, X)+b)

    for i in range (A.shape[1]):
        
        # Convert probability to prediction (0, 1)
        A[A > 0.5] = 1
        A[A <= 0.5] = 0
        Y_prediction = A

        return Y_prediction
