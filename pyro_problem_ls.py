import numpy as np
from linear_approximation import LinearApproximation


class PyroProblemLS:

    rho = 7500
    c = 1.15e11
    a = 0
    e = 15.1
    chi = 5.62e-9
    cv = 420
    _lambda = 2.1
    pi = -2.12e-4
    alpha = 3.13e-5
    z = 5e-12
    temperature_0 = 293

    def __init__(self, length, n, time_period, nt, t_0):
        self.length = length
        self.n = n
        self.timePeriod = time_period
        self.nt = nt
        self.dt = time_period / nt
        self.t_0 = t_0
        self.approximation = LinearApproximation(length, n)
        self._calculate_matrices_by_bilinear_forms()
        self.U_0 = np.zeros(n + 1)
        self.U_dot_0 = np.zeros(n + 1)
        self.P_0 = np.zeros(n + 1)
        self.Theta_0 = np.zeros(n + 1)
        self.Q_0 = np.zeros(n + 1)
        self.L_sigma = np.zeros((n + 1, nt + 1))
        self.L_e = np.zeros((n + 1, nt + 1))
        self.L_theta = np.zeros((n + 1, nt + 1))
        self.L_q = np.zeros((n + 1, nt + 1))
        nx = (self.nt + 1) / 11.2
        for i in range(1, self.nt + 1):
            if i <= nx:
                self.L_q[0, i] = 293 * (i / nx) * (1.0 / PyroProblemLS.temperature_0)
            else:
                self.L_q[0, i] = 293 * (1.0 / PyroProblemLS.temperature_0)

    def _calculate_matrices_by_bilinear_forms(self):
        self.M = self.approximation.get_phi_phi_matrix(PyroProblemLS.rho)
        self.A = self.approximation.get_der_der_matrix(PyroProblemLS.a)
        self.C = self.approximation.get_der_der_matrix(PyroProblemLS.c)

        self.E = self.approximation.get_der_der_matrix(-PyroProblemLS.e)
        self.B = self.approximation.get_phi_der_matrix(PyroProblemLS.c * PyroProblemLS.alpha)
        self.CHI = self.approximation.get_der_der_matrix(PyroProblemLS.chi)

        self.Z = self.approximation.get_der_der_matrix(PyroProblemLS.z)
        self.PI = self.approximation.get_phi_der_matrix(-PyroProblemLS.pi)
        self.S = self.approximation.get_phi_phi_matrix(PyroProblemLS.rho * PyroProblemLS.cv / PyroProblemLS.temperature_0)
        self.K_classic = self.approximation.get_der_der_matrix(PyroProblemLS._lambda / PyroProblemLS.temperature_0)

        self.kappa = 1.0 / (PyroProblemLS.temperature_0 * PyroProblemLS._lambda)
        self.K = self.approximation.get_phi_phi_matrix(self.kappa)
        self.W = self.approximation.get_der_phi_matrix(1.0 / PyroProblemLS.temperature_0)
