
import numpy as np


class HLL_Euler_Eq_Solver:

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

    def primitive_variables(self, U):
        # Recover velocity and sound speed from the conserved variables.
        v = U[1] / U[0]
        p = (self.gamma - 1) * (U[2] - ((U[1])**2) / (2 * U[0]))
        c = np.sqrt(self.gamma * p / U[0])

        return v, c

    def wave_speeds(self, UL, UR):
        # Estimate the left- and right-going wave speeds
        # needed for the HLL flux.
        vL, cL = self.primitive_variables(UL)
        vR, cR = self.primitive_variables(UR)

        SL = min(vL - cL, vR - cR)
        SR = max(vL + cL, vR + cR)

        return [SL, SR]

    def hll_flux(self, UL, UR):
        # Calculate the numerical flux using the HLL approximation.
        SL, SR = self.wave_speeds(UL, UR)

        if SL >= 0:
            F = self.F(UL)
        elif SR <= 0:
            F = self.F(UR)
        elif SL <= 0 and SR >= 0:
            F = (
                SR * self.F(UL)
                - SL * self.F(UR)
                + SL * SR * (UR - UL)
            ) / (SR - SL)

        return F

    def time_step(self, U):
        U_new = U.copy()

        # MUSCL reconstruction followed by the HLL flux
        # at the cell interfaces.
        for i in range(2, len(U) - 2):

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

            FL = self.hll_flux(ULL, ULR)
            FR = self.hll_flux(URL, URR)

            U_new[i] = U[i] - (self.dt / self.dx) * (FR - FL)

        # Apply boundary conditions using the nearest interior states.
        U_new[0] = U_new[1] = U_new[2]
        U_new[-1] = U_new[-2] = U_new[-3]

        return U_new

    def time_step_spherical(self, U):
        # Add the spherical source term after the finite-volume update.
        U_1 = self.time_step(U)
        U_2 = U_1 + (self.dt / 2) * self.S(U_1)
        U_new = U_1 + self.dt * self.S(U_2)

        return U_new

