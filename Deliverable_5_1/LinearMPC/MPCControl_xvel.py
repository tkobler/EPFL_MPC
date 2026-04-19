import numpy as np

from .MPCControl_base import MPCControl_base
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib as plt


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # =============================================================
        #                   LQR Weights
        # =============================================================
        Q = 100*np.eye(3)
        R = np.eye(1) 

        # =============================================================
        #                   Constraints Polyhedrons
        # =============================================================
        # u in U = { u | Mu <= m }
        M = np.array([[1], [-1]])
        m = np.array([0.26, 0.26])
        U = Polyhedron.from_Hrep(M, m)
        
        # x in X = { x | Fx <= f }
        F = np.array([[0,1,0], 
                      [0,-1,0]])  
        f = np.array([0.1745, 0.1745])
        X = Polyhedron.from_Hrep(F, f)
        
        # =============================================================
        #                       Invariant Sets
        #   (Not maximal invariant because target not know a priori)
        # =============================================================
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K
        
        A_cl = self.A+self.B@K
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        O_inf = self.max_invariant_set(A_cl, X.intersect(KU))
        
        # =============================================================
        #                   Optimization Variables
        # =============================================================
        nx, nu = self.nx, self.nu
        self.x_var = cp.Variable((nx, self.N + 1), name='x')
        self.u_var = cp.Variable((nu, self.N), name='u')
        self.x0_var = cp.Parameter((nx,), name='x0')
        self.eps_min = cp.Parameter((len(f),self.N), name='eps_min')
        # initializes target xs and us (steady state values)
        self.x_ss = cp.Parameter((nx,), name='x_ss')
        self.u_ss = cp.Parameter((nu,), name='u_ss')
        # Helps for dimensions later
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
            cost += cp.quad_form(self.u_var[:,i]-self.u_ss, R)

        # Terminal cost
        cost += cp.quad_form(self.x_var[:, -1]-self.x_ss, Qf)
        
        # =============================================================
        #                  Constraints with slack variables
        # =============================================================
        constraints = []

        # Initial condition
        constraints.append((self.x_var[:, 0]) == (self.x0_var))

        # System dynamics
        constraints.append((self.x_var[:, 1:] - x_ss_mat[:, 1:]) == 
                           self.A @ (self.x_var[:, :-1] - x_ss_mat[:, :-1]) 
                           + self.B @ (self.u_var - u_ss_mat))
        
        # State constraints F @ X <= f + epsilon_min
        # Here this is constrained over the whole horizon (except for the terminal constraint)
        # So eps_min is also of size (2, N) 
        constraints.append(X.A @ (self.x_var[:, :-1]) <= X.b.reshape(-1, 1) + self.eps_min)
        
        # Input constraints M @ U <= m
        constraints.append(U.A @ (self.u_var) <= U.b.reshape(-1, 1))

        # Terminal Constraints
        # We did not inclue the slack variable in the terminal constraint
        constraints.append(O_inf.A @ (self.x_var[:, -1]-self.x_ss)  <= O_inf.b.reshape(-1, 1))
        
        # =============================================================
        #                  Definition of the Problem
        # =============================================================
        # Main MPC
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        
        # Optimization to find slack variable
        super()._setup_controller(F, f, M, m)

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
        
        #! The values returned here are already in global frame (not deltas)
        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)
        # YOUR CODE HERE
        #################################################
        
        return u0, x_traj, u_traj
