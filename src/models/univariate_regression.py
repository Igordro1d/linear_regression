import math

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_w = w * x[i] + b
        cost += (f_w - y[i]) ** 2
    total_cost = 1/(2*m) * cost
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_w = w * x[i] + b
        dj_dw += (f_w - y[i]) * x[i]
        dj_db += (f_w - y[i])
    final_dj_dw = 1/m * dj_dw
    final_dj_db = 1/m * dj_db
    return final_dj_dw, final_dj_db

def gradient_descent(x, y, w_init, b_init, alpha, num_iters, cost_function, gradient_function):
    w = w_init
    b = b_init
    parameter_history = []
    cost_history = []
    for i in range(num_iters):
        dj_dw , dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % math.ceil(num_iters/10) == 0:
            parameter_history.append((w, b))
            cost_history.append(cost_function(x, y, w, b))
            print(f"Iteration {i:4}: Cost = {cost_history[-1]:0.3e}",
                  f"dj_dw: {dj_dw:0.3e}, dj_db: {dj_db:0.3e}",
                  f"w: {w:0.3e}, b: {b:0.3e}")
    return w, b, parameter_history, cost_history


