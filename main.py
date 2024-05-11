from dynamics import time_evolution
from plotting import plot_hysteresis
from constants import OMEGA, DT, H_AMP
import numpy as np
import time

def main():
    start_time = time.time()
    frequency = OMEGA[0]
    periods = np.arange(0, 2 * np.pi / frequency, DT)
    magnetizations, applied_fields = time_evolution(frequency, periods)  # Capture both returns
    M_z = [m[0][2] for m in magnetizations]  # Extract M_z component from magnetizations
    H_app = applied_fields  # Already calculated as the second output of time_evolution

    plot_hysteresis(M_z, H_app)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
