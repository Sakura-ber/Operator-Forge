import numpy as np
import random

def compute_derivatives(u, dx, dy):
    ux, uy = np.gradient(u, dx, dy)
    uxx, uxy = np.gradient(ux, dx, dy)
    _, uyy = np.gradient(uy, dx, dy)
    return ux, uy, uxx, uyy, uxy

def calculate_sobolev_norm(u, dx, dy, k, p):
    total_sum = np.sum(np.abs(u)**p)
    if k >= 1:
        ux, uy = np.gradient(u, dx, dy)
        total_sum += np.sum(np.abs(ux)**p + np.abs(uy)**p)
    if k >= 2:
        ux, uy = np.gradient(u, dx, dy)
        uxx, uxy = np.gradient(ux, dx, dy)
        _, uyy = np.gradient(uy, dx, dy)
        total_sum += np.sum(np.abs(uxx)**p + 2*np.abs(uxy)**p + np.abs(uyy)**p)
    norm_val = (total_sum * dx * dy)**(1/p)
    return norm_val

def apply_robin_boundary(u, X, Y, alpha, beta, geometry='rect'):
    if geometry == 'rect':
        dist = np.minimum(np.minimum(X, 1-X), np.minimum(Y, 1-Y))
    else: # circle
        r = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
        dist = np.maximum(0, 0.5 - r)
    weight = np.tanh(dist * 10) 
    return u * weight

def generate_base(X, Y):
    type_ = random.choice(['power', 'sin', 'exp', 'log', 'constant', 'polynomial'])
    var = random.choice(['X', 'Y'])
    v = X if var == 'X' else Y
    if type_ == 'power':
        return v ** random.uniform(0.5, 2)
    elif type_ == 'sin':
        return np.sin(random.randint(1, 2) * np.pi * v)
    elif type_ == 'exp':
        return np.exp(random.uniform(-0.5, 0.5) * v)
    elif type_ == 'log':
        return np.log(1.1 + random.uniform(0.1, 1) * v)
    elif type_ == 'constant':
        return np.full_like(X, random.uniform(-1, 1))
    else: # polynomial
        return random.uniform(-0.5, 0.5)*v**2 + random.uniform(-0.5, 0.5)*v

def build_G(X, Y):
    ops = [lambda a,b: a+b, lambda a,b: a-b, lambda a,b: a*b]
    funcs = [generate_base(X, Y) for _ in range(3)]
    res = funcs[0]
    for f in funcs[1:]:
        op = random.choice(ops)
        res = op(res, f)
    return res