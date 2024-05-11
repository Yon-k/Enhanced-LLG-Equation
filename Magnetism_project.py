import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import time


gamma_0 = 2.21 * 10 ** 5
alpha = 0.05
mu_0 = 4 * np.pi * 10 ** (-7)
M_s = 1.787 / mu_0
R = 8 * 10 ** (-9)
V = (4 / 3) * np.pi * R ** 3
K = 5 * 10 ** 5
omega = [10 ** 8]
H_amp = 1.5 / mu_0
dt = 10 ** (-12)

#--Constants to calculate alpha:--
e = 1.602 * 10 ** (-19)
h_bar = (6.626 * 10 **(-34))/(2*np.pi)
m_e = 9.10938356 * 10 ** (-31)
c = 299792458

# --------------------------------------------------- MATH. FUNCTIONS -------------------------------------------------

def susceptibility(M, dH):
    if len(M) == 1:
        return 0
    else:
        backward_diff = (M[-1][2] - M[-2][2]) / dH
    return 1/backward_diff


def susceptibility_vector(M, H):
    chi = []
    for i, M_i in enumerate(M):
        if i == 0:
            chi.append((M[i+1]-M[i])/(H[i+1]-H[i]))
        elif i == (len(M)-1):
            chi.append((M[i]-M[i-1])/(H[i]-H[i-1]))
        else:
            chi.append((M[i+1]-M[i-1])/(2*(H[i+1]-H[i-1])))  #Central diff
    return chi


def fNNP_2(t, y, freq, full_vector):
    M, n = y
    eq = []
    H_ani = (2 * K / (mu_0 * M_s)) * np.dot(M, n) * n
    H = np.array([0, 0, H_amp * np.sin(freq * t)]) + H_ani
    #constants = e*h_bar*mu_0*M_s/(8*(m_e**2)*(c**2))
    #alpha = e*h_bar*mu_0*M_s/(8*(m_e**2)*(c**2)) * (1 + susceptibility([M[0] for M in full_vector], dt))
    eq.append(-(1 / (1 + alpha ** 2)) * gamma_0 * np.cross(M, H) -
              (1 / (1 + alpha ** 2)) * (alpha * gamma_0) * np.cross(M, np.cross(M, H)))
    eq.append(np.array([0, 0, 0]))
    return np.array(eq)


# ----------------------------------------------------- FUNCTIONS -----------------------------------------------------

def RK6(x0, y0, f, dx, freq, full_vector):
    k1 = dx * f(x0, y0, freq, full_vector)
    k2 = dx * f(x0 + (1 / 5) * dx, y0 + (1 / 5) * k1, freq, full_vector)
    k3 = dx * f(x0 + (3 / 10) * dx, y0 + (3 / 40) * k1 + (9 / 40) * k2, freq, full_vector)
    k4 = dx * f(x0 + (3 / 5) * dx, y0 + (3 / 10) * k1 - (9 / 10) * k2 + (6 / 5) * k3, freq, full_vector)
    k5 = dx * f(x0 + dx, y0 - (11 / 54) * k1 + (5 / 2) * k2 - (70 / 27) * k3 + (35 / 27) * k4, freq, full_vector)
    k6 = dx * f(x0 + (7 / 8) * dx,
                y0 + (1631 / 55296) * k1 + (175 / 512) * k2 + (575 / 13824) * k3 + (44275 / 110592) * k4 + (
                            253 / 4096) * k5, freq, full_vector)
    return y0 + (37 / 378) * k1 + (250 / 621) * k3 + (125 / 594) * k4 + (512 / 1771) * k6

def Time_evolution(freq, period, M_0=None, n_0=None):
    np.random.seed(46482610)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    if M_0 is None:
        M_0 = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    if n_0 is None:
        n_0 = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    # --Initial conditions--
    y = [np.array([M_0, n_0])]
    H_z = [0]
    for time in period[1:]:
        H_ani = (2 * K / (mu_0 * M_s)) * np.dot(y[-1][0], n_0) * n_0
        H_z.append(H_amp * np.sin(freq * time) + H_ani[2])
        y.append(RK6(time, y[-1], fNNP_2, dt, freq, y))
    return y, H_z

def Hysteresis_loop(freq, period):
    n_0 = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)])
    y, H_z = Time_evolution(freq=freq, period=period, M_0=n_0, n_0=n_0)
    return [[magnetiz[0][0], magnetiz[0][1], magnetiz[0][2]] for magnetiz in y], H_z


# ------------------------------------------------------- PLOTS -------------------------------------------------------

def Hysteresis_plot(x, y, i, period_size, loops):
    #for k in range(loops):
    #    if k == loops-2:
    #        plt.plot(x[k * period_size:(k * period_size) + period_size],
    #             y[k * period_size:(k * period_size) + period_size], label=f'$\omega = 10^{i} rad/s$')
    #    if k == loops-1:
    #        plt.plot(x[k * period_size:(k * period_size) + period_size],
    #                 y[k * period_size:(k * period_size) + period_size], label=f'$\omega = 10^{i} rad/s$')
    plt.plot(x, y)# label=f'$\omega = 10^{i} rad/s$')
    plt.xlabel('$H_{applied_z}/H_k$')
    plt.ylabel('$M_z/M_s$')
    plt.title('Fixed easy axis: n=($1/\sqrt{2}$, 0 , $1/\sqrt{2}$)')

def M_evolution_plot(x, y, z):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='M(t) curve')
    ax.legend()
    ax.set_xlabel('M_x/M_s')
    ax.set_ylabel('M_y/M_s')
    ax.set_zlabel('M_z/M_s')
    plt.savefig('M(t)')
    plt.show()

def susceptibility_plot(M , H, period_size, loops):
    chi = np.array(susceptibility_vector(M, H))
    for k in range(loops):
        if k == 0:
            plt.plot(H[(k * period_size):((k * period_size) + period_size)],
                     chi[k * period_size:(k * period_size) + period_size], label=k + 1)
        #if k == loops-1:
        #    plt.plot(H[(k * period_size):((k * period_size) + period_size)],
        #             chi[k * period_size:(k * period_size) + period_size], label=k + 1)

    plt.plot(np.array(H) / (2 * K / (mu_0 * M_s)), chi)
    plt.xlabel('$H_{applied_z}/H_k$')
    plt.ylabel('$\chi$')
    plt.savefig('susceptibility_vector')
    plt.tight_layout()
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

start_time = time.time()
if __name__ == '__main__':
    N_loops = 5
    # Plot_1: Hysteresis loop
    i = 8
    for w_i in omega:
        print(f'iteration of w_i={w_i}')
        period = np.arange(0, N_loops * (2 * np.pi / w_i), dt)
        H_app = [H_amp * np.sin(w_i * time) for time in period]
        M_vector, H_z = Hysteresis_loop(freq=w_i, period=period)  #H_z takes into account the anisotropy field
        M_z = [Mz[2] for Mz in M_vector]
        a = int(((N_loops-1)/N_loops) * len(period))
        period_size = len(np.arange(0, 2 * np.pi / w_i, dt))
        Hysteresis_plot(np.array(H_app[a:]) / (2 * K / (mu_0 * M_s)), M_z[a:], i, period_size, N_loops)
        #susceptibility_plot(M_s * np.array(M_z), H_app, len(M_z), N_loops)
        i += 1
    print("--- %s minuts ---" % (time.time() / 60 - start_time / 60))
    plt.legend()
    plt.savefig('Hysteresis_loop')
    plt.show()


    # Plot2: Magnetization plot
    M_x = [Mx[0] for Mx in M_vector]
    M_y = [My[1] for My in M_vector]
    M_z = [Mz[2] for Mz in M_vector]
    M_evolution_plot(M_x, M_y, M_z)

    # Plot3: Hz(t) vs Mz(t)
    plt.plot(period, M_z, label='M_z')
    plt.plot(period, np.array(H_app) / H_amp, label='H_z')
    plt.tight_layout()
    plt.xlabel('time [s]')
    plt.legend()
    plt.savefig('Time_M_H')
    plt.show()

    # Plot4: susceptibility
    susceptibility_plot(M_s*np.array(M_z), H_app, len(H_app), N_loops)
