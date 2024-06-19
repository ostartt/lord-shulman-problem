import numpy as np
import os

from scipy.linalg import solve_banded
from pyro_problem_ls import PyroProblemLS


class OSRecurrentSchemeLS:

    def __init__(self, length, n, time_period, nt, t_0, gamma, beta):
        self.length = length
        self.n = n
        self.timePeriod = time_period
        self.nt = nt
        self.dt = time_period / nt
        self.t_0 = t_0
        self.problem = PyroProblemLS(length, n, time_period, nt, t_0)
        self.gamma = gamma
        self.beta = beta

        self.q_scale = (1.0 / self.problem.t_0)
        self.p_scale = 1e-6
        self.u_scale = 1
        self.theta_scale = 1

        self._calculate_big_matrix_for_time_integration(gamma, beta)
        self._init_result_arrays()

    def _init_result_arrays(self):
        self.U = np.zeros((self.n + 1, self.nt + 1))
        self.U_dot = np.zeros_like(self.U)
        self.U_dot_dot = np.zeros_like(self.U)

        self.P = np.zeros_like(self.U)
        self.P_dot = np.zeros_like(self.U)

        self.Theta = np.zeros_like(self.U)
        self.Theta_dot = np.zeros_like(self.U)
        self.Theta_dot_dot = np.zeros_like(self.U)

        self.Q = np.zeros_like(self.U)
        self.Q_dot = np.zeros_like(self.U)

        self.electric_field = np.zeros((self.n, self.nt + 1))
        self.mechanical_stress = np.zeros((self.n, self.nt + 1))

    def _calculate_big_matrix_for_time_integration(self, gamma, beta):
        self.total_matrix = np.zeros((4 * (self.n + 1), 4 * (self.n + 1)))
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                self.total_matrix[4 * i, 4 * j] = self.u_scale * self.u_scale * (self.problem.M[i, j]
                                                                                 + self.dt * gamma * self.problem.A[
                                                                                     i, j]
                                                                                 + self.dt * self.dt * beta *
                                                                                 self.problem.C[i, j])
                self.total_matrix[4 * i, 4 * j + 1] = - self.u_scale * self.p_scale * self.dt * gamma * self.problem.E[
                    j, i]
                self.total_matrix[4 * i, 4 * j + 2] = - self.theta_scale * self.u_scale * self.dt * gamma * \
                                                      self.problem.B[j, i]
                self.total_matrix[4 * i, 4 * j + 3] = 0

                self.total_matrix[4 * i + 1, 4 * j] = self.u_scale * self.p_scale * (
                        self.dt * gamma * self.problem.E[i, j])
                self.total_matrix[4 * i + 1, 4 * j + 1] = (self.p_scale * self.p_scale *
                                                           (self.problem.CHI[i, j]
                                                            + self.dt * gamma * self.problem.Z[i, j]))
                self.total_matrix[4 * i + 1, 4 * j + 2] = self.theta_scale * self.p_scale * self.problem.PI[j, i]
                self.total_matrix[4 * i + 1, 4 * j + 3] = 0

                self.total_matrix[4 * i + 2, 4 * j] = self.theta_scale * self.u_scale * self.dt * gamma * \
                                                      self.problem.B[i, j]
                self.total_matrix[4 * i + 2, 4 * j + 1] = self.theta_scale * self.p_scale * self.problem.PI[i, j]
                self.total_matrix[4 * i + 2, 4 * j + 2] = self.theta_scale * self.theta_scale * self.problem.S[i, j]
                self.total_matrix[4 * i + 2, 4 * j + 3] = self.theta_scale * self.q_scale * self.dt * gamma * \
                                                          self.problem.W[j, i]

                self.total_matrix[4 * i + 3, 4 * j] = 0
                self.total_matrix[4 * i + 3, 4 * j + 1] = 0
                self.total_matrix[4 * i + 3, 4 * j + 2] = -self.q_scale * self.theta_scale * self.dt * gamma * \
                                                          self.problem.W[i, j]
                self.total_matrix[4 * i + 3, 4 * j + 3] = (self.q_scale * self.q_scale *
                                                           (self.problem.t_0 * self.problem.K[i, j]
                                                            + self.dt * gamma * self.problem.K[i, j]))
        self._convert_to_banded_matrix(7, 7)

    def _convert_to_banded_matrix(self, lower_bandwidth, upper_bandwidth):
        n = self.total_matrix.shape[0]
        self.banded_matrix = np.zeros((1 + lower_bandwidth + upper_bandwidth, n))

        for i in range(n):
            for j in range(max(0, i - lower_bandwidth), min(n, i + upper_bandwidth + 1)):
                self.banded_matrix[lower_bandwidth + i - j, j] = self.total_matrix[i, j]

    def _calculate_electric_field(self):
        for i in range(0, self.n - 1):
            for j in range(0, self.nt + 1):
                self.electric_field[i, j] = -(self.p_scale *
                                              (self.problem.approximation.get_phi_der()[0] * self.P[i, j]
                                               + self.problem.approximation.get_phi_der()[1] * self.P[i + 1, j]))

    def _calculate_mechanical_stress(self):
        for i in range(0, self.n - 1):
            for j in range(0, self.nt + 1):
                self.mechanical_stress[i, j] = (
                        self.problem.c * self.u_scale * (self.problem.approximation.get_phi_der()[0] * self.U[i, j]
                                                         + self.problem.approximation.get_phi_der()[1] * self.U[
                                                             i + 1, j])
                        - 0.5 * self.problem.c * self.problem.alpha * self.theta_scale * (
                                self.Theta[i, j] + self.Theta[i + 1, j])
                        + self.p_scale * self.problem.e *
                        (self.problem.approximation.get_phi_der()[0] * self.P[i, j]
                         + self.problem.approximation.get_phi_der()[1] * self.P[i + 1, j]))

    def _apply_initial_conditions(self):
        self.U[:, 0] = self.problem.U_0
        self.U_dot[:, 0] = self.problem.U_dot_0
        self.U_dot_dot[:, 0] = 0

        self.P[:, 0] = self.problem.P_0
        self.P_dot[:, 0] = 0

        self.Theta[:, 0] = self.problem.Theta_0
        self.Theta_dot[:, 0] = 0
        self.Theta_dot_dot[:, 0] = 0

        self.Q[:, 0] = self.problem.Q_0
        self.Q_dot[:, 0] = 0

    def _apply_boundary_conditions_for_time_step(self, i):
        nx = (self.nt + 1) / 11.2
        if i <= nx:
            self.Theta[0, i] = 293 * (i / nx) / self.theta_scale
            self.Theta_dot[0, i] = 293 * 1e12 / self.theta_scale
        else:
            self.Theta[0, i] = 293 / self.theta_scale
            self.Theta_dot[0, i] = 0
        self.Theta[self.n, i] = 0
        self.Q[self.n, i] = 0

    def _total_effective_force_for_time_step(self, i):
        F_u = (self.problem.L_sigma[:, i]
               - self.u_scale * (self.problem.A + self.dt * self.gamma * self.problem.C) @ self.U_dot[:, i]
               - self.u_scale * self.problem.C @ self.U[:, i]
               + self.p_scale * self.problem.E.T @ self.P[:, i]
               + self.theta_scale * self.problem.B.T @ self.Theta[:, i])
        F_p = (self.problem.L_e[:, i] - self.u_scale * self.problem.E @ self.U_dot[:, i]
               - self.p_scale * self.problem.Z @ self.P[:, i])
        F_theta = (self.problem.L_theta[:, i] - self.u_scale * self.problem.B @ self.U_dot[:, i]
                   - self.q_scale * self.problem.W.T @ self.Q[:, i])
        F_q = (self.problem.L_q[:, i] + self.theta_scale * self.problem.W @ self.Theta[:, i]
               - self.q_scale * self.problem.K @ self.Q[:, i])

        total_F = np.zeros(4 * self.n + 4)
        for j in range(0, self.n + 1):
            total_F[4 * j] = self.u_scale * F_u[j]
            total_F[4 * j + 1] = self.p_scale * F_p[j]
            total_F[4 * j + 2] = self.theta_scale * F_theta[j]
            total_F[4 * j + 3] = self.q_scale * F_q[j]
        return total_F

    def _calculate_next_nodal_values_for_time_step(self, i):
        self.U[:, i] = self.U[:, i - 1] + self.dt * self.U_dot[:, i - 1] \
                       + (0.5 * self.dt ** 2) * self.U_dot_dot[:, i - 1]
        self.U_dot[:, i] = self.U_dot[:, i - 1] + self.dt * self.U_dot_dot[:, i - 1]
        self.P[:, i] = self.P[:, i - 1] + self.dt * self.P_dot[:, i - 1]
        self.Theta[:, i] = self.Theta[:, i - 1] + self.dt * self.Theta_dot[:, i - 1]
        self.Q[:, i] = self.Q[:, i - 1] + self.dt * self.Q_dot[:, i - 1]

    def perform_os_recurrent_scheme(self):
        self._apply_initial_conditions()
        for i in range(1, self.nt + 1):
            self._calculate_next_nodal_values_for_time_step(i)

            total_F = self._total_effective_force_for_time_step(i)
            total_result = solve_banded((7, 7), self.banded_matrix, total_F)
            for j in range(0, self.n + 1):
                self.U_dot_dot[j, i] = total_result[4 * j]
                self.P_dot[j, i] = total_result[4 * j + 1]
                self.Theta_dot[j, i] = total_result[4 * j + 2]
                self.Q_dot[j, i] = total_result[4 * j + 3]

            self._apply_boundary_conditions_for_time_step(i)

        self._calculate_electric_field()
        self._calculate_mechanical_stress()

    def save_results(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        np.savetxt(folder_name + "/" + "u.txt", self.u_scale * self.U)
        np.savetxt(folder_name + "/" + "p.txt", self.p_scale * self.P)
        np.savetxt(folder_name + "/" + "theta.txt", self.theta_scale * self.Theta)
        np.savetxt(folder_name + "/" + "q.txt", self.q_scale * self.Q)

        np.savetxt(folder_name + "/" + "electric_field.txt", self.electric_field)
        np.savetxt(folder_name + "/" + "mechanical_stress.txt", self.mechanical_stress)
