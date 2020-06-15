import numpy as np

def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0

    # check
    assert(w.shape == (dim, 1))

    return w, b
