import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def g(t, c, a):
    return np.exp(-np.pow(a - c * t, 2)) + np.cos(3 * (a - c * t))

def f(x):
    return np.exp(-np.pow(x, 2)) + np.cos(3 * x)

def run_lax_friedrichs(m, CFL=0.5, T=1, c=2, a=0, b=1):
    dx = (b - a) / (m - 1)
    xs = np.linspace(a, b, m)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u_ghost = np.zeros(m + 1)
    u_ghost[:-1] = f(xs)
    for j in range(nt):
        u_ghost[0] = g(j * dt, c, a)
        u_ghost[-1] = 2 * u_ghost[-2] - u_ghost[-3]
        u_ghost[1:-1] = (1 / 2) * (u_ghost[2:] + u_ghost[:-2]) - (nu / 2) * (u_ghost[2:] - u_ghost[:-2])
    
    u = u_ghost[:-1]
    u[0] = g(T, c, a)
    u_exact = f(xs - c * T)
    inf_error = np.max(np.abs(u - u_exact))
    sq_error = np.sqrt(np.sum((u - u_exact)**2) * dx)
    return xs, u, u_exact, dx, inf_error, sq_error


def get_L2(u, dx, c):
    l = np.zeros(np.shape(u)[0])
    l[1:-1] = -c * (u[2:] - u[:-2]) / (2 * dx)
    return l

def get_K2(u_0, t, c, a, dx):
    u = u_0.copy()
    u[0] = g(t, c, a)
    u[-1] = 2 * u[-2] - u[-3]
    return get_L2(u, dx, c)

def run_rk4cds_ghost(m, CFL=0.5, T=1, c=2, a=0, b=1):
    dx = (b - a) / (m - 1)
    xs = np.linspace(a, b, m)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u_ghost = np.zeros(m + 1)
    u_ghost[:-1] = f(xs)
    for j in range(nt):
        k1 = get_K2(u_ghost, j * dt, c, a, dx)
        k2 = get_K2(u_ghost + 0.5 * dt * k1, (j + 0.5) * dt, c, a, dx)
        k3 = get_K2(u_ghost + 0.5 * dt * k2, (j + 0.5) * dt, c, a, dx)
        k4 = get_K2(u_ghost + dt * k3, (j + 1) * dt, c, a, dx)
        u_ghost = u_ghost + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
    u = u_ghost[:-1]
    u[0] = g(T, c, a)
    u_exact = f(xs - c * T)
    inf_error = np.max(np.abs(u - u_exact))
    sq_error = np.sqrt(np.sum((u - u_exact)**2) * dx)
    return xs, u, u_exact, dx, inf_error, sq_error


def get_L_one_side(u, dx, c):
    l = np.zeros(np.shape(u)[0])
    l[1:-1] = -c * (u[2:] - u[:-2]) / (2 * dx)
    l[-1] = -c * (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * dx)
    return l

def get_K_one_side(u_0, t, c, a, dx):
    u = u_0.copy()
    u[0] = g(t, c, a)
    return get_L_one_side(u, dx, c)

def run_rk4cds_one_side(m, CFL=0.5, T=1, c=2, a=0, b=1):
    dx = (b - a) / (m - 1)
    xs = np.linspace(a, b, m)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u = f(xs)
    for j in range(nt):
        k1 = get_K_one_side(u, j * dt, c, a, dx)
        k2 = get_K_one_side(u + 0.5 * dt * k1, (j + 0.5) * dt, c, a, dx)
        k3 = get_K_one_side(u + 0.5 * dt * k2, (j + 0.5) * dt, c, a, dx)
        k4 = get_K_one_side(u + dt * k3, (j + 1) * dt, c, a, dx)
        u = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
    u[0] = g(T, c, a)
    u_exact = f(xs - c * T)
    inf_error = np.max(np.abs(u - u_exact))
    sq_error = np.sqrt(np.sum((u - u_exact)**2) * dx)
    return xs, u, u_exact, dx, inf_error, sq_error


def get_L4(u, dx, c):
    l = np.zeros(np.shape(u)[0])
    l[2:-2] = -c * (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[:-4]) / (12 * dx)
    return l

def get_K4(u_0, t, c, a, dx):
    u = u_0.copy()
    u[1] = g(t, c, a)
    u[0] = 5 * u[1] - 10 * u[2] + 10 * u[3] - 5 * u[4] + u[5]
    u[-2] = 5 * u[-3] - 10 * u[-4] + 10 * u[-5] - 5 * u[-6] + u[-7]
    u[-1] = 15 * u[-3] - 40 * u[-4] + 45 * u[-5] - 24 * u[-6] + 5 * u[-7]
    return get_L4(u, dx, c)

def run_rk4c4_ghost(m, CFL=0.5, T=1.0, c=2, a=0, b=1):
    dx = (b - a) / (m - 1)
    xs = np.linspace(a, b, m)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u_ghost = np.zeros(m + 3)
    u_ghost[1:-2] = f(xs)
    for j in range(nt):
        k1 = get_K4(u_ghost, j * dt, c, a, dx)
        k2 = get_K4(u_ghost + 0.5 * dt * k1, (j + 0.5) * dt, c, a, dx)
        k3 = get_K4(u_ghost + 0.5 * dt * k2, (j + 0.5) * dt, c, a, dx)
        k4 = get_K4(u_ghost + dt * k3, (j + 1) * dt, c, a, dx)
        u_ghost = u_ghost + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
    u = u_ghost[1:-2]
    u[0] = g(T, c, a)
    u_exact = f(xs - c * T)
    inf_error = np.max(np.abs(u - u_exact))
    sq_error = np.sqrt(np.sum((u - u_exact)**2) * dx)
    inf_norm = np.max(np.abs(u))
    sq_norm = np.sqrt(np.sum(u**2) * dx)
    return xs, u, u_exact, dx, inf_error, sq_error, inf_norm, sq_norm

if __name__ == "__main__":
    # You can test the 4 schemes with any parameters you want here.
    xs, u, u_exact, dx, inf_error, sq_error, inf_norm, sq_norm = run_rk4c4_ghost(1001)
    print("l2_norm of the error: ", sq_error)