import numpy as np


class LinearApproximation:

    def __init__(self, length, n):
        self.length = length
        self.n = n
        self.h = length / n

    def get_phi_phi_matrix(self, coefficient):
        J = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n):
            J[i, i] += coefficient * self.h / 3.0
            J[i, i + 1] += coefficient * self.h / 6.0
            J[i + 1, i] += coefficient * self.h / 6.0
            J[i + 1, i + 1] += coefficient * self.h / 3.0
        return J

    def get_der_der_matrix(self, coefficient):
        A = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n):
            A[i, i] += coefficient / self.h
            A[i, i + 1] += -coefficient / self.h
            A[i + 1, i] += -coefficient / self.h
            A[i + 1, i + 1] += coefficient / self.h
        return A

    def get_phi_der_matrix(self, coefficient):
        K = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n):
            K[i, i] += -0.5 * coefficient
            K[i, i + 1] += 0.5 * coefficient
            K[i + 1, i] += -0.5 * coefficient
            K[i + 1, i + 1] += 0.5 * coefficient
        return K

    def get_der_phi_matrix(self, coefficient):
        G = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n):
            G[i, i] += -0.5 * coefficient
            G[i, i + 1] += -0.5 * coefficient
            G[i + 1, i] += 0.5 * coefficient
            G[i + 1, i + 1] += 0.5 * coefficient
        return G

    def get_phi_der(self):
        return [-1.0 / self.h, 1.0 / self.h]
