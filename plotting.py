import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_hysteresis(M, H):
    plt.figure(figsize=(8, 6))
    plt.plot(H, M, label='Hysteresis loop')
    plt.xlabel('$H_{applied_z}/H_k$')
    plt.ylabel('$M_z/M_s$')
    plt.title('Hysteresis Loop')
    plt.legend()
    plt.show()
