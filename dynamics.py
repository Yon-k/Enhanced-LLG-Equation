import numpy as np
from constants import *
from math_utils import cross_product, dot_product
from tqdm import tqdm

def effective_field(t, M, n, freq):
    H_ani = (2 * K / (MU_0 * M_S)) * dot_product(M, n) * n
    H = np.array([0, 0, H_AMP * np.sin(freq * t)]) + H_ani
    return H

def dM_dt(t, M, n, freq):
    H = effective_field(t, M, n, freq)
    term1 = -(1 / (1 + ALPHA**2)) * GAMMA_0 * cross_product(M, H)
    term2 = -(1 / (1 + ALPHA**2)) * (ALPHA * GAMMA_0) * cross_product(M, cross_product(M, H))
    return term1 + term2

def RK6(t, y, dt, freq):
    M, n = y
    f = lambda t, M, n: dM_dt(t, M, n, freq)
    
    k1 = dt * f(t, M, n)
    k2 = dt * f(t + 1/5*dt, M + 1/5*k1, n)
    k3 = dt * f(t + 3/10*dt, M + 3/40*k1 + 9/40*k2, n)
    k4 = dt * f(t + 3/5*dt, M + 3/10*k1 - 9/10*k2 + 6/5*k3, n)
    k5 = dt * f(t + dt, M - 11/54*k1 + 5/2*k2 - 70/27*k3 + 35/27*k4, n)
    k6 = dt * f(t + 7/8*dt, M + 1631/55296*k1 + 175/512*k2 + 575/13824*k3 + 44275/110592*k4 + 253/4096*k5, n)

    M_new = M + (37/378*k1 + 250/621*k3 + 125/594*k4 + 512/1771*k6)
    return M_new, n  # Assuming 'n' remains constant

def time_evolution(freq, periods, initial_conditions=None):
    np.random.seed(46482610)
    theta, phi = np.random.uniform(0, np.pi), np.random.uniform(0, 2 * np.pi)
    M_0 = initial_conditions or np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    n_0 = M_0.copy()
    magnetizations = [(M_0, n_0)]
    applied_fields = [0]  # Initialize with zero or appropriate initial value

    print("Starting time evolution...")
    for time in tqdm(periods[1:], desc="Computing time evolution"):
        new_state = RK6(time, magnetizations[-1], DT, freq)
        magnetizations.append(new_state)
        applied_fields.append(effective_field(time, new_state[0], new_state[1], freq)[2])

    print("Time evolution completed.")
    return magnetizations, applied_fields


