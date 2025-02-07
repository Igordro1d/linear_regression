import math
import numpy as np

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0

    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        cost += (f_wb - y[i]) ** 2
    total_cost = 1/(2*m) * cost
    return total_cost

def compute_gradient(X, y, w, b):
    #number of examples, number of features
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        dj_db += (f_wb - y[i])
        for j in range(n):
            dj_dw[j] += (f_wb - y[i]) * X[i][j]
    final_dj_dw = 1/m * dj_dw
    final_dj_db = 1/m * dj_db
    return final_dj_dw, final_dj_db

def gradient_descent(X, y, w_init, b_init, cost_function, gradient_function, alpha, num_iters):
    w = w_init
    b = b_init
    cost_history = []
    for i in range(num_iters):
        dj_dw , dj_db = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % math.ceil(num_iters/10) == 0:
            cost_history.append(cost_function(X, y, w, b))
            print(f"Iteration {i:4}: Cost = {cost_history[-1]:0.3e}")
    return w, b, cost_history