import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sbp
import scipy.sparse as sp

atol=1e-10

def g(t, c, a):
    return np.exp(-np.pow(a - c * t, 2)) + np.cos(13 * (a - c * t))

def f(x):
    return np.exp(-np.pow(x, 2)) + np.cos(13 * x)

def get_u_t(u, t, c, a, D, Hinv):
    du_dt = - c * D.dot(u)
    e0 = np.zeros(np.shape(u))
    e0[0] = 1
    penalty = - c * Hinv @ e0 * (u[0] - g(t, c, a))
    du_dt += penalty
    return du_dt

def run_rk6(sbp_func, m, CFL=0.5, T=1, c=1.1, a=-1, b=1):
    dx = (b - a) / (m - 1)
    xs = np.linspace(a, b, m)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    
    u = f(xs)
    D, H, Hinv = sbp_func(m, dx)
    B = np.zeros((m, m))
    B[0, 0] = -1
    B[m - 1, m - 1] = 1

    LHS = H @ D + D.T.conj() @ H
    hold = np.linalg.norm(LHS.toarray() - B) < atol
    k = 0
    while(np.linalg.norm(D.dot(xs**k) - k * xs**(k - 1)) < atol):
        k += 1
    sbp_order = k

    for j in range(nt):
        k1 = get_u_t(u, j * dt, c, a, D, Hinv)
        k2 = get_u_t(u + dt * k1 / 6, j * dt + dt / 6, c, a, D, Hinv)
        k3 = get_u_t(u + dt * (k1 / 12 + k2 / 12), j * dt + dt / 6, c, a, D, Hinv)
        k4 = get_u_t(u + dt * (- 4 * k2 / 33 + 5 * k3 / 11), j * dt + dt / 3, c, a, D, Hinv)
        k5 = get_u_t(u + dt * (- k1 / 4 - 29 * k2 / 44 + 31 * k3 / 22), j * dt + dt / 2, c, a, D, Hinv)
        k6 = get_u_t(u + dt * (3 * k1 / 11 + 8 * k2 / 33 - 4 * k3 / 11 + k4 / 11 + 14 * k5 / 33), j * dt + 2 * dt / 3, c, a, D, Hinv)
        k7 = get_u_t(u + dt * (- 17 * k1 / 48 - 5 * k2 / 12 + k3 + k4 - 13 * k5 / 12 + 11 * k6 / 16), j * dt + 5 * dt / 6, c, a, D, Hinv)
        k8 = get_u_t(u + dt * (20 * k1 / 39 + 12 * k2 / 39 - 31 * k3 / 39 - k4 / 39 + 34 * k5 / 39 - 11 * k6 / 39 + 16 * k7 / 39), j * dt + dt, c, a, D, Hinv)
        u = u + dt * (13 * k1 / 200 + 4 * k3 / 25 + 11 * k4 / 40 + 11 * k6 / 40 + 4 * k7 / 25 + 13 * k8 / 200)
            
    u_exact = f(xs - c * T)
    inf_error = np.max(np.abs(u - u_exact))
    sq_error = np.sqrt(np.sum((u - u_exact)**2) * dx)
    return xs, u, u_exact, dx, inf_error, sq_error, hold, sbp_order

if __name__ == "__main__":
    # You can test RK6 with 4 SBP operators here.
    xs, u, u_exact, dx, inf_error, sq_error, hold, sbp_order = run_rk6(sbp.sbp21, 100)
    print("l2_norm of the error: ", sq_error)
