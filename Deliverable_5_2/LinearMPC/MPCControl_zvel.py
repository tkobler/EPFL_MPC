import numpy as np

from .MPCControl_base import MPCControl_base
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib as plt
from typing import Tuple


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

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
        
        # x in X = {x | Fx <= f}, no constraints → 0 rows
        F = np.zeros((0, self.nx))  # shape (0,1)
        f = np.zeros(0)             # shape (0,)
        X = Polyhedron.from_Hrep(F, f)
        
        # =============================================================
        #                       Invariant Sets
        #   (Not maximal invariant because target not know a priori)
        # =============================================================
        _, Qf, _ = dlqr(self.A, self.B, Q, R)
        
        # =============================================================
        #                   Optimization Variables
        # =============================================================
        nx, nu = self.nx, self.nu
        self.x_var = cp.Variable((nx, self.N + 1), name='x')
        self.u_var = cp.Variable((nu, self.N), name='u')
        self.x0_var = cp.Parameter((nx,), name='x0')

        # Variables for disturbance and reference tracking
        self.nd = 1
        self.ny = 1
                
        self.x_ss = cp.Parameter((nx,), name='x_ss')
        self.u_ss = cp.Parameter((nu,), name='u_ss')
        ones_Np1 = np.ones((1, self.N + 1))  # (1, N+1)
        ones_N   = np.ones((1, self.N))      # (1, N)
        x_ss_mat = cp.reshape(self.x_ss, (nx, 1)) @ ones_Np1
        u_ss_mat = cp.reshape(self.u_ss, (nu, 1)) @ ones_N

        self.setup_estimator()
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
        #                   Constraints without slack variables
        # =============================================================
        constraints = []

        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_var)

        # System dynamics
        constraints.append((self.x_var[:, 1:] - x_ss_mat[:, 1:]) == 
                           self.A @ (self.x_var[:, :-1] - x_ss_mat[:, :-1]) 
                           + self.B @ (self.u_var - u_ss_mat))

        # No constraints on states
        
        # Input constraints
        constraints.append(U.A @ (self.u_var) <= U.b.reshape(-1, 1))

        # No terminal constraints  
        
        # =============================================================
        #                  Definition of the Problem
        # =============================================================
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)


    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        
        # =============================================================
        #                  Compute steady state targets
        # =============================================================
        """ Here [I-A  -B][xs] = [Bd * dk]
                 [C     0][us]   [r-Cd*dk] , With r = x_target, C = 1, and Bd = B
                 
        => xs = x_target (eq. 1), and (I-A)xs - B*us = B*dk
        This gives: us = B^(-1)(I-A)xs-B*dk.
        
        However A*x_target + B*u_target = x_target => B^(-1)(I-A)xs = u_target
        
        So: us = u_target - dk (eq. 2)
                 """
        self.x_ss.value = x_target # (eq. 1)
        self.u_ss.value = u_target - self.d_estimate # (eq. 2)

        # =============================================================
        #                           Solve
        # =============================================================
        # the solver is in the mother class as it is the same for all controllers
        #! The values returned here are already in global frame (not deltas)
        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)

        # =============================================================
        #            Observer update and disturbance estimation
        # =============================================================
        
        # the estimator needs the delta variables
        self.update_estimator(x0-self.x_ss.value, u0-u_target)
        
        # YOUR CODE HERE
        #################################################
        return u0, x_traj, u_traj

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        # Define disturbance and output dimensions
        
        self.C = np.eye(1)  # Output matrix
        self.Bd = self.B
        self.Cd = np.array([[0]])
        
        # A_hat = [A  Bd;  0  I]
        self.A_hat = np.vstack((
            np.hstack((self.A, self.Bd)),
            np.hstack((np.zeros((self.nd, self.nx)), np.eye(self.nd)))
            ))
        
        # B_hat = [B; 0]
        self.B_hat = np.vstack((self.B, np.zeros((self.nd, self.nu))))

        # C_hat = [C  Cd]
        self.C_hat = np.hstack((self.C, self.Cd))

        # initialize with 0
        self.d_estimate = np.zeros((self.nd,)) 

        poles = np.array([0.5, 0.6])
        from scipy.signal import place_poles
        res = place_poles(self.A_hat.T, self.C_hat.T, poles)
        
        # L = K^T for observer
        self.d_gain = -res.gain_matrix.T  

        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        yk = self.C @ x_data
        if not hasattr(self, 'x_hat_prev'):
            self.x_hat_prev = np.hstack((x_data, self.d_estimate))
        y_hat = self.C_hat @ self.x_hat_prev  # Predicted output
        
        # Observer update with innovation (measurement - prediction)
        x_hat_next = self.A_hat @ self.x_hat_prev + self.B_hat @ u_data + self.d_gain @ (y_hat - yk)
        
        # Store for next iteration
        self.x_hat_prev = x_hat_next
        self.d_estimate = x_hat_next[self.nx:]
        # YOUR CODE HERE
        ##################################################
