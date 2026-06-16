# The experiements and discussions are in another report file. This file only contains the implementations of the 8 schemes.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_circulant

# Initial Condition
def g(x):
    return np.exp(1j * x)

# Upwind
def run_upwind(nx, CFL, T, c=1.0, a=-np.pi, b=np.pi):
    dx = (b - a) / nx
    xs = np.linspace(a, b, nx, endpoint=False)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u = g(xs)
    for _ in range(nt):
        if c > 0:
            u = u - nu * (u - np.roll(u, 1))
        else:
            u = u - nu * (np.roll(u, -1) - u)
            
    u_exact = g(a + np.mod(xs - c * T - a, b - a))
    error = np.max(np.abs(u - u_exact))
    return xs, u, u_exact, dx, error

# Crank-Nicholson
def run_cn(nx, CFL, T, c=1.0, a=-np.pi, b=np.pi):
    dx = (b - a) / nx
    xs = np.linspace(a, b, nx, endpoint=False)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt
    
    u = np.zeros(nx, dtype="complex")
    v = np.zeros(nx, dtype="complex")
    nu = (c / 4) * (dt / dx)

    s = np.zeros(nx)
    s[-1] = nu
    s[0] = 1.0
    s[1] = -nu

    u = g(xs)

    for j in range(nt):
        v = u - nu * (np.roll(u, -1) - np.roll(u, 1))
        u = solve_circulant(s, v)

    u_exact = g(a + np.mod(xs - c * T - a, b - a))
    error = np.max(np.abs(u - u_exact))
    return xs, u, u_exact, dx, error

# Lax-Wendroff
def run_lw(nx, CFL, T, c=1.0, a=-np.pi, b=np.pi):
    dx = (b - a) / nx
    xs = np.linspace(a, b, nx, endpoint=False)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u = np.zeros(nx, dtype="complex")
    u = g(xs)

    for j in range(nt):
        u = u - (nu / 2) * (np.roll(u, -1) - np.roll(u, 1)) + (nu**2 / 2) * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
            
    u_exact = g(a + np.mod(xs - c * T - a, b - a))
    error = np.max(np.abs(u - u_exact))
    return xs, u, u_exact, dx, error

# Leapfrog
def run_lf(nx, CFL, T, c=1.0, a=-np.pi, b=np.pi):
    dx = (b - a) / nx
    xs = np.linspace(a, b, nx, endpoint=False)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u1 = np.zeros(nx, dtype="complex")
    u1 = g(xs)
    u2 = u1 - nu * (u1 - np.roll(u1, 1))

    for j in range(0, nt - 1):
        temp = u2
        u2 = -nu * (np.roll(u2, -1) - np.roll(u2, 1)) + u1
        u1 = temp
            
    u_exact = g(a + np.mod(xs - c * T - a, b - a))
    error = np.max(np.abs(u2 - u_exact))
    return xs, u2, u_exact, dx, error

# FTCDS
def run_ftcds(nx, CFL, T, c=1.0, a=-np.pi, b=np.pi):
    dx = (b - a) / nx
    xs = np.linspace(a, b, nx, endpoint=False)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u = g(xs)
    for j in range(0, nt - 1):
        u = u - (nu / 2) * (np.roll(u, -1) - np.roll(u, 1))
            
    u_exact = g(a + np.mod(xs - c * T - a, b - a))
    error = np.max(np.abs(u - u_exact))
    return xs, u, u_exact, dx, error

# Lax-Friendichs
def run_laxf(nx, CFL, T, c=1.0, a=-np.pi, b=np.pi):
    dx = (b - a) / nx
    xs = np.linspace(a, b, nx, endpoint=False)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u = g(xs)
    for j in range(nt):
        u = (1 / 2) * (np.roll(u, -1) + np.roll(u, 1)) - (nu / 2) * (np.roll(u, - 1) - np.roll(u, 1))
            
    u_exact = g(a + np.mod(xs - c * T - a, b - a))
    error = np.max(np.abs(u - u_exact))
    return xs, u, u_exact, dx, error

# RK2CD2
def run_rk2(nx, CFL, T, c=1.0, a=-np.pi, b=np.pi):
    dx = (b - a) / nx
    xs = np.linspace(a, b, nx, endpoint=False)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx
    
    u = g(xs)
    for j in range(0, nt):
        d = np.roll(u, -1) - np.roll(u, 1)
        u = u - (nu / 2) * d + (nu ** 2 / 8) * (np.roll(d, -1) - np.roll(d, 1))
            
    u_exact = g(a + np.mod(xs - c * T - a, b - a))
    error = np.max(np.abs(u - u_exact))
    return xs, u, u_exact, dx, error

# Helper Function for RK4CD4
def L(u, dx):
    return (-np.roll(u, -2) + 8*np.roll(u, -1) - 8*np.roll(u, 1) + np.roll(u, 2)) / (12 * dx)

# RK4CD4
def run_rk4(nx, CFL, T, c=1.0, a=-np.pi, b=np.pi):
    dx = (b - a) / nx
    xs = np.linspace(a, b, nx, endpoint=False)
    dt = CFL * dx / abs(c)
    nt = int(np.ceil(T / dt))
    dt = T / nt 
    nu = c * dt / dx

    u = np.zeros(nx, dtype="complex")
    u = g(xs)

    for _ in range(nt):
        k1 = -c * L(u, dx)
        k2 = -c * L(u + 0.5 * dt * k1, dx)
        k3 = -c * L(u + 0.5 * dt * k2, dx)
        k4 = -c * L(u + dt * k3, dx)
        u = u + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
    u_exact = g(a + np.mod(xs - c * T - a, b - a))
    error = np.max(np.abs(u - u_exact))
    return xs, u, u_exact, dx, error

# You can test the 8 schemes with any parameters you want.
if __name__ == "__main__":
    _, _, _, _, error_up = run_upwind(nx=1000, CFL=0.8, T=12.5)
    _, _, _, _, error_cn = run_cn(nx=1000, CFL=0.8, T=12.5)
    _, _, _, _, error_lw = run_lw(nx=1000, CFL=0.8, T=12.5)
    _, _, _, _, error_lf = run_lf(nx=1000, CFL=0.8, T=12.5)
    _, _, _, _, error_ftcds = run_ftcds(nx=1000, CFL=0.8, T=12.5)
    _, _, _, _, error_laxf = run_laxf(nx=1000, CFL=0.8, T=12.5)
    _, _, _, _, error_rk2 = run_rk2(nx=1000, CFL=0.8, T=12.5)
    _, _, _, _, error_rk4 = run_rk4(nx=1000, CFL=0.8, T=12.5)
    print("Error of Upwind scheme: \t\t", error_up)
    print("Error of Crank-Nicholson scheme: \t", error_cn)
    print("Error of Leap-Wendroff scheme: \t\t", error_lw)
    print("Error of Leapfrog scheme: \t\t", error_lf)
    print("Error of FTCDS scheme: \t\t\t", error_ftcds)
    print("Error of Lax-Friendichs scheme: \t", error_laxf)
    print("Error of RK2CD2 scheme: \t\t", error_rk2)
    print("Error of RK4CD4 scheme: \t\t", error_rk4)