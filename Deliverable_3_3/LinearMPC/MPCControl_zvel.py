import numpy as np

from .MPCControl_base import MPCControl_base
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib as plt


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # =============================================================
        #                   LQR Weights
        # =============================================================
        Q = 100*np.eye(1)
        R = np.eye(1) 

        # =============================================================
        #                   Constraints Polyhedrons
        # =============================================================
        # u in U = { u | Mu <= m }
        M = np.array([[1], [-1]])
        m = np.array([80, -40])
        U = Polyhedron.from_Hrep(M, m)
        
        # x in X = { x | Fx <= f }
        # x in X = {x | Fx <= f}, no constraints → 0 rows
        F = np.zeros((0, self.nx))  # shape (0,1)
        f = np.zeros(0)             # shape (0,)
        X = Polyhedron.from_Hrep(F, f)
        
        # =============================================================
        #                       Invariant Sets
        #   (Not maximal invariant because target not know a priori)
        # =============================================================
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K
        
        A_cl = self.A+self.B@K
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        O_inf = self.max_invariant_set(A_cl, X.intersect(KU)) # eske le X vaut 0 ou RN dans ce cas
        
        # =============================================================
        #                   Optimization Variables
        # =============================================================
        nx, nu = self.nx, self.nu
        self.x_var = cp.Variable((nx, self.N + 1), name='x')
        self.u_var = cp.Variable((nu, self.N), name='u')
        self.x0_var = cp.Parameter((nx,), name='x0')

        # initializes target xss and uss (steady state values)
        self.x_ss = cp.Parameter((nx,), name='x_ss')
        self.u_ss = cp.Parameter((nu,), name='u_ss')
        
        # expand x_ss, u_ss as matrices over the horizon
        ones_Np1 = np.ones((1, self.N + 1))  # (1, N+1)
        ones_N   = np.ones((1, self.N))      # (1, N)

        # x_ss_mat: shape (nx, N+1), each column = x_ss
        x_ss_mat = cp.reshape(self.x_ss, (nx, 1)) @ ones_Np1
        # u_ss_mat: shape (nu, N), each column = u_ss
        u_ss_mat = cp.reshape(self.u_ss, (nu, 1)) @ ones_N

        # =============================================================
        #                   Cost Function
        # =============================================================
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(self.x_var[:,i]-self.x_ss, Q)
            cost += cp.quad_form(self.u_var[:,i] - self.u_ss, R)

        # Terminal cost
        cost += cp.quad_form(self.x_var[:, -1]-self.x_ss, Qf)

        # =============================================================
        #                  Constraints
        # =============================================================
        constraints = []

        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_var)

        # System dynamics
        constraints.append((self.x_var[:, 1:] - x_ss_mat[:, 1:]) == 
                           self.A @ (self.x_var[:, :-1] - x_ss_mat[:, :-1]) 
                           + self.B @ (self.u_var - u_ss_mat))
        
        # State constraints
        # No constraints on states
        
        # Input constraints
        constraints.append(U.A @ (self.u_var) <= U.b.reshape(-1, 1))

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
        
        # adds the tracking to the problem
        self.x_ss.value = x_target   
        self.u_ss.value = u_target
        
        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE

        self.d_estimate = ...
        self.d_gain = ...

        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        self.d_estimate = ...
        # YOUR CODE HERE
        ##################################################
