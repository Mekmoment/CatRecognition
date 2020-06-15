import numpy as np

def sigmoid(z):
    "Calculate sigmoid of z"
    s = 1/(1+np.exp(-z))
    return s