import numpy as np
from os_recurrent_scheme_ls import OSRecurrentSchemeLS

if __name__ == "__main__":
    L = 1e-8
    T = 1.12e-11
    n = 128
    nt = 1200
    t = np.linspace(0, T, nt + 1)
    gamma = 0.5
    beta = 0.25
    t_0 = 1e-12

    results_folder = "results"
    scheme1 = OSRecurrentSchemeLS(L, n, T, nt, 1e-11, gamma, beta)
    scheme1.perform_os_recurrent_scheme()
    scheme1.save_results(results_folder + "/10_11")

    scheme2 = OSRecurrentSchemeLS(L, n, T, nt, 1e-12, gamma, beta)
    scheme2.perform_os_recurrent_scheme()
    scheme2.save_results(results_folder + "/10_12")

    scheme3 = OSRecurrentSchemeLS(L, n, T, nt, 1e-14, gamma, beta)
    scheme3.perform_os_recurrent_scheme()
    scheme3.save_results(results_folder + "/10_14")
