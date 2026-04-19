import numpy as np

from .MPCControl_base import MPCControl_base
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib.pyplot as plt


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    # Compute minimal robust invariant set
    def min_robust_invariant_set(self, A_cl: np.ndarray, W: Polyhedron, max_iter: int = 30) -> Polyhedron:
        nx = A_cl.shape[0]
        Omega = W
        itr = 0
        A_cl_ith_power = np.eye(nx)
        while itr < max_iter:
            A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
            Omega_next = Omega + A_cl_ith_power @ W
            Omega_next.minHrep()#inplace=True)  # optionally: Omega_next.minVrep()
            if np.linalg.matrix_norm(A_cl_ith_power, ord=2) < (1e-1):
                print('Minimal robust invariant set computation converged after {0} iterations.'.format(itr))
                break

            if itr == max_iter:
                print('Minimal robust invariant set computation did NOT converge after {0} iterations.'.format(itr))
            
            Omega = Omega_next
            itr += 1
        if itr == max_iter:
            print('Minimal robust invariant set computation did NOT converge after {0} iterations.'.format(itr))
        return Omega_next

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # =============================================================
        #                   LQR Weights
        # =============================================================
        Q = np.diag([300, 1000])
        R = np.eye(1)*10
        
        Qterminal = np.diag([300, 1000])
        Rterminal = np.eye(1)*10

        # =============================================================
        #                   Constraints Polyhedrons
        # =============================================================
        # u in U = { u | Mu <= m }
        M = np.array([[1], [-1]])
        m = np.array([80, -40])
        U = Polyhedron.from_Hrep(M, m)
        
        # x in X = { x | Fx <= f }
        F = np.array([[0, -1]])  
        f = np.array([0])
        X = Polyhedron.from_Hrep(F, f)

        # w in W = { w | HBw <= hB }
        H = np.vstack((np.eye(2), -np.eye(2)))
        h = np.array([5*self.B[0,0],5*self.B[1,0], 15*self.B[0,0], 15*self.B[1,0]])
        self.W = Polyhedron.from_Hrep(H, h)

        # =============================================================
        #                   Min Robust Invariant Set
        # =============================================================
        K, Qf, _ = dlqr(self.A, self.B, Qterminal, Rterminal)
        
        K_inv, Qf_inv, _ = dlqr(self.A, self.B, Q, R)
        
        self.K_inv = -K_inv
        
        self.K = -K
        A_cl = self.A + self.B @ self.K_inv
        A_cl_terminal = self.A + self.B @ self.K

        E = self.min_robust_invariant_set(A_cl, self.W, 200)
        
        #! PLOTTING E
        ax = plt.figure().gca()
        ax.set_title("Invariant set (2D projection) of E" f", {self.__class__.__name__}")
        E.plot(ax)
        #! PLOTTING E
        
        # =============================================================
        #                   Constraints Tightening
        # =============================================================
        # tightened state constraints
        X_tilde = X - E
        # tightened input constraints
        U_tilde = U - self.K_inv @ E
        
        # =============================================================
        #                       Invariant Sets
        #   (Not maximal invariant because target not know a priori)
        # =============================================================
        # terminal set for tube MPC
        KU_tilde = Polyhedron.from_Hrep(U_tilde.A @ self.K, U_tilde.b - U_tilde.A @ self.us)
        X_decale = Polyhedron.from_Hrep(X_tilde.A, X_tilde.b - X_tilde.A @ self.xs)
        Xf_tilde = self.max_invariant_set(A_cl_terminal,  X_decale.intersect(KU_tilde),150)
        
        print("Normal input constraints: ",U.V)
        print("Tightend input constraints: ",U_tilde.V)
        
        # =============================================================
        #                   Optimization Variables
        # =============================================================
        nx, nu = self.nx, self.nu
        self.z_var = cp.Variable((self.N+1, nx), name='z') # warning: devinition is transposed from other MPC controllers
        self.v_var = cp.Variable((self.N, nu), name='v')
        self.x0_var = cp.Parameter((nx,), name='x0')
                
        ones_Np1 = np.ones((1, self.N + 1))  # (1, N+1)
        ones_N   = np.ones((1, self.N))      # (1, N)
        xs_mat = cp.reshape(self.xs, (nx, 1)) @ ones_Np1
        us_mat = cp.reshape(self.us, (nu, 1)) @ ones_N

        # =============================================================
        #                   Cost Function
        # =============================================================
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(self.z_var[i,:] - self.xs, Q)
            cost += cp.quad_form(self.v_var[i,:] - self.us, R)

        # Terminal cost
        cost += cp.quad_form(self.z_var[self.N, :] - self.xs, Qf)

        # =============================================================
        #                   Constraints without slack variables
        # =============================================================
        constraints = []
        
        # initial condition: x0 in z0 + E
        constraints.append(E.A @ (self.x0_var - self.z_var[0]) <= E.b)

        # System dynamics
        constraints.append((self.z_var[1:].T - xs_mat[:, :-1]) == 
                           self.A @ (self.z_var[:-1].T - xs_mat[:, :-1]) 
                           + self.B @ (self.v_var.T - us_mat)) #delta

        # state constraints
        constraints.append(X_tilde.A @ self.z_var[:-1].T <= X_tilde.b.reshape(-1, 1))

        # input constraints
        constraints.append(U_tilde.A @ self.v_var[:-1].T <= U_tilde.b.reshape(-1, 1))

        # terminal set
        constraints.append(Xf_tilde.A @ (self.z_var[-1].T - xs_mat[:, -1]) <= Xf_tilde.b.reshape(-1, 1)) #delta

        # =============================================================
        #                  Definition of the Problem
        # =============================================================
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)


    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        self.x0_var.value = x0
        self.ocp.solve(solver=cp.PIQP)#, **settings)
        assert self.ocp.status == cp.OPTIMAL
        z = self.z_var.value
        v = self.v_var.value

        u_traj = np.zeros((self.N, 1))
        x_traj = np.zeros((self.N + 1, self.nx))
        x_traj[0, :] = x0
        for k in range(self.N):
            #N Note: here we reconstruct the actual trajectories from the nominal ones,
            # insted of returning the nominal ones (tube centers) directly,
            # this is done as to keep cosistency with other MPC controllers
            u_traj[k] = v[k] + self.K_inv @ (x_traj[k, :] - z[k])
            xk = self.A @ x_traj[k, :] + self.B @ u_traj[k]
            x_traj[k+1, :] = xk
        u0 = v[0] + self.K_inv @ (x0 - z[0])
        # YOUR CODE HERE
        #################################################

        return u0, x_traj.T, u_traj.T

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