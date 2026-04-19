import numpy as np

from .MPCControl_base import MPCControl_base
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib as plt


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # =============================================================
        #                   LQR Weights
        # =============================================================
        Q = 10000*np.eye(2)
        R = 0.1*np.eye(1) 

        # =============================================================
        #                   Constraints Polyhedrons
        # =============================================================
        # u in U = { u | Mu <= m }
        M = np.array([[1], [-1]])
        m = np.array([20, 20]) 
        U = Polyhedron.from_Hrep(M, m)
        
        # x in X = {x | Fx <= f}, no constraints → 0 rows
        F = np.zeros((0, self.nx))
        f = np.zeros(0)
        X = Polyhedron.from_Hrep(F, f)

        # =============================================================
        #                   Max Invariant Sets
        # =============================================================
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K
        
        A_cl = self.A+self.B@K
        KU = Polyhedron.from_Hrep(U.A @ K, U.b - U.A @ self.us)
        O_inf = self.max_invariant_set(A_cl, KU)
        
        # =============================================================
        #                   Optimization Variables
        # =============================================================
        nx, nu = self.nx, self.nu
        self.x_var = cp.Variable((nx, self.N + 1), name='x')
        self.u_var = cp.Variable((nu, self.N), name='u')
        self.x0_var = cp.Parameter((nx,), name='x0')

        xs_col = np.asarray(self.xs).reshape(nx, 1)
        us_col = np.asarray(self.us).reshape(nu, 1)
        xs_mat = xs_col @ np.ones((1, self.N + 1))   # (nx, N+1)
        us_mat = us_col @ np.ones((1, self.N))       # (nu, N)

        # =============================================================
        #                   Cost Function
        # =============================================================
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(self.x_var[:,i] - self.xs, Q)
            cost += cp.quad_form(self.u_var[:,i] - self.us, R)
            
        # Terminal cost
        cost += cp.quad_form(self.x_var[:, -1] - self.xs, Qf)

        # =============================================================
        #                  Constraints
        # =============================================================
        constraints = []

        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_var)

        # System dynamics
        constraints.append((self.x_var[:,1:]- xs_mat[:,:-1]) ==
                           self.A @ (self.x_var[:,:-1] - xs_mat[:,:-1]) 
                           + self.B @ (self.u_var - us_mat))

        # State constraints
        # No constraints on states

        # Input constraints
        constraints.append(U.A @ self.u_var <= U.b.reshape(-1, 1))

        # Terminal Constraints
        # No terminal constraints   

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj 
