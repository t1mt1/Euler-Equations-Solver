
import numpy as np
import matplotlib.pyplot as plt


class Rusanov_Euler_Eq_Solver:

    def __init__(self, dt=None, dx=None, gamma=None):
        self.dt = dt
        self.dx = dx
        self.gamma = gamma

    def F(self, U_i):
        # Euler flux for the conserved variables:
        # density, momentum, and total energy.
        rho, mom, E = U_i
        v = mom / rho
        p = (self.gamma - 1) * (E - (mom**2) / (2 * rho))

        return np.array([mom, mom * v + p, v * (E + p)])

    def S(self, U):
        # Geometrical source term for the spherical Euler equations.
        rho = U[:, 0]
        mom = U[:, 1]
        E = U[:, 2]

        v = mom / rho
        v[0] = 0

        p = (self.gamma - 1) * (E - (mom**2) / (2 * rho))
        r = (np.arange(len(U)) + 0.5) * self.dx

        return (-2 / r[:, None]) * np.column_stack([
            rho * v,
            rho * v**2,
            v * (E + p)
        ])

    def minmod(self, a, b):
        # Minmod slope limiter used in the MUSCL reconstruction.
        return np.where(
            a * b <= 0,
            0.0,
            np.where(np.abs(a) < np.abs(b), a, b)
        )

    def time_step(self, U):
        # Calculate primitive variables and the local speed of sound.
        v = U[:, 1] / U[:, 0]
        p = (self.gamma - 1) * (
            U[:, 2] - ((U[:, 1])**2) / (2 * U[:, 0])
        )
        c = np.sqrt(self.gamma * p / U[:, 0])

        U_new = U.copy()

        # MUSCL reconstruction and Rusanov flux at cell interfaces.
        for i in range(2, len(U) - 2):
            aL = max(
                abs(v[i - 1]) + c[i - 1],
                abs(v[i]) + c[i]
            )
            aR = max(
                abs(v[i + 1]) + c[i + 1],
                abs(v[i]) + c[i]
            )

            ULL = U[i - 1] + 0.5 * self.minmod(
                U[i - 1] - U[i - 2],
                U[i] - U[i - 1]
            )
            ULR = U[i] - 0.5 * self.minmod(
                U[i] - U[i - 1],
                U[i + 1] - U[i]
            )
            URL = U[i] + 0.5 * self.minmod(
                U[i] - U[i - 1],
                U[i + 1] - U[i]
            )
            URR = U[i + 1] - 0.5 * self.minmod(
                U[i + 1] - U[i],
                U[i + 2] - U[i + 1]
            )

            FL = 0.5 * (
                self.F(ULL) + self.F(ULR) - aL * (ULR - ULL)
            )
            FR = 0.5 * (
                self.F(URR) + self.F(URL) - aR * (URR - URL)
            )

            U_new[i] = U[i] - (self.dt / self.dx) * (FR - FL)

        # Apply boundary conditions using the first and last
        # reconstructed interior states.
        U_new[0] = U_new[1] = U_new[2]
        U_new[-1] = U_new[-2] = U_new[-3]

        return U_new

    def time_step_spherical(self, U):
        # Apply the finite-volume update first, then integrate
        # the spherical source term using a two-stage method.
        U_1 = self.time_step(U)
        U_2 = U_1 + (self.dt / 2) * self.S(U_1)
        U_new = U_1 + self.dt * self.S(U_2)

        return U_new



