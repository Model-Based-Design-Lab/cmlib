import numpy as np

def matPower(m, n):
    return np.linalg.matrix_power(m, n)

def printMatrix(m):
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, linewidth=100000, threshold=100000)
    print(m)