import numpy as np
def z_score_normalisation(X):
    #axis 0 means to operate down each column
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma

def minmax_scaling(X):
    return (X - X.min()) / (X.max() - X.min())