import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
from matplotlib import pyplot as plt


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self, hx, kx, hu, ku) -> None: 
        #################################################
        #YOUR CODE HERE

        # =============================================================
        #                   Optimization weights
        # =============================================================
        S = 200*np.eye(len(kx))
        s = 0.1*np.ones(len(kx))

        # =============================================================
        #                  Constraints for slack variable
        # =============================================================
        # (Same constraints as for the main problem, but with epsilon >= 0)
        # u in U = { u | Mu <= m }
        U = Polyhedron.from_Hrep(hu, ku)
        # x in X = { x | Fx <= f }
        X = Polyhedron.from_Hrep(hx, kx)

        # =============================================================
        #                   Optimization Variables
        # =============================================================
        nx, nu = self.nx, self.nu
        self.u_var_eps = cp.Variable((nu, self.N), name='u')
        self.x0_var_eps = cp.Parameter((nx,), name='x0')
        self.x_var_eps = cp.Variable((nx, self.N + 1), name='x')
        self.eps_var = cp.Variable((len(kx), self.N + 1), name='eps', nonneg=True)

        # initializes target xss and uss and helpful matrixes
        ones_Np1 = np.ones((1, self.N + 1))  # (1, N+1)
        ones_N   = np.ones((1, self.N))      # (1, N)
        x_ss_mat = cp.reshape(self.x_ss, (nx, 1)) @ ones_Np1
        u_ss_mat = cp.reshape(self.u_ss, (nu, 1)) @ ones_N

        # =============================================================
        #                          Cost Function
        # =============================================================
        # cost function: eps.T @ S @ eps + s.T @ eps
        cost = 0
        for i in range(self.N + 1):
            cost += cp.quad_form(self.eps_var[:,i], S)
            cost += s @ self.eps_var[:,i]

        # =============================================================
        #                          Constraint definition
        # =============================================================
        constraints = []

        # Initial condition
        constraints.append((self.x_var_eps[:, 0]) == (self.x0_var_eps))

        # System dynamics
        constraints.append((self.x_var_eps[:, 1:] - x_ss_mat[:, 1:]) == 
                           self.A @ (self.x_var_eps[:, :-1] - x_ss_mat[:, :-1]) 
                           + self.B @ (self.u_var_eps - u_ss_mat))
        
        # State constraints
        constraints.append(X.A @ (self.x_var_eps[:, :-1]) <= X.b.reshape(-1, 1) + self.eps_var[:, :-1])
        
        # Input constraints
        constraints.append(U.A @ (self.u_var_eps) <= U.b.reshape(-1, 1))

        # eps >= 0
        constraints.append(self.eps_var >= 0)

        self.ocp_eps = cp.Problem(cp.Minimize(cost), constraints)
        

    def max_invariant_set(self, A_cl, X: Polyhedron, max_iter = 100) -> Polyhedron:
        """
        Compute invariant set for an autonomous linear time invariant system x^+ = A_cl x
        """
        O = X
        itr = 1
        converged = False
        while itr < max_iter:
            Oprev = O
            F, f = O.A, O.b
            # Compute the pre-set
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
            O.minHrep(True)
            _ = O.Vrep 
            if O == Oprev:
                converged = True
                break
            itr += 1
        
        if converged:
            # for plotting purposes
            print(
                f"Maximum invariant set successfully computed after {itr} iterations. "
                f"(from class {self.__class__.__name__})"
                )
            if A_cl.shape[1] >= 3:
                # ax = plt.figure().gca()
                # ax.set_title("Projections of invariant set (dims 0–1 and 1–2)" f", {self.__class__.__name__}")
                # O.projection(dims=(0,1)).plot(ax)
                # O.projection(dims=(1,2)).plot(ax)
                pairs = [((1,0), r'$(\theta,\ \omega)$'),
                ((3,2), r'$(p,\ v)$'),
                ((1,3), r'$(\theta,\ p)$'),
                ((1,2), r'$(\theta,\ v)$')]

                fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                fig.suptitle(f"Invariant set projections, {self.__class__.__name__}")

                for ax, (dims, label) in zip(axs.ravel(), pairs):
                    O.projection(dims=dims).plot(ax)
                    ax.set_title(label)
                    ax.grid(True)

                plt.tight_layout()
                plt.show()
            elif A_cl.shape[1] >= 2:
                ax = plt.figure().gca()
                ax.set_title("Invariant set (2D projection)" f", {self.__class__.__name__}")
                O.plot(ax)
            elif A_cl.shape[1] == 1:
                #check for empty set
                if O.V.shape[0] > 0:
                    a1 = O.A[0,0]
                    a2 = O.A[1,0]

                    b1 = O.b[0]
                    b2 = O.b[1]

                    #print("x ", ">" if a1 < 0 else "<", " ", b1/a1)
                    #print("x ", ">" if a2 < 0 else "<", " ", b2/a2)

                    #creates a simple 1d plot
                    ax = plt.figure().gca()
                    ax.set_title("Invariant set (1D projection)" f", {self.__class__.__name__}")
                    x_vals = np.linspace(-10,10,100)
                    y_vals = np.zeros_like(x_vals)
                    ax.plot(x_vals, y_vals, color='blue')
                    ax.axvline(x=b1/a1, color='red', linestyle='--')
                    ax.axvline(x=b2/a2, color='red', linestyle='--')
                    ax.fill_betweenx([-1,1], min(b1/a1, b2/a2), max(b1/a1, b2/a2), color='red', alpha=0.3)
                    plt.ylim(-1,1)  
                else:
                    #creates a simple 1d plot at xs
                    ax = plt.figure().gca()
                    ax.set_title("Invariant set (1D projection)" f", {self.__class__.__name__}")
                    x_vals = np.linspace(-10,10,100)
                    y_vals = np.zeros_like(x_vals)
                    ax.plot(x_vals, y_vals, color='blue')
                    ax.axvline(x=self.xs, color='red', linestyle='--')
                    plt.ylim(-1,1)  
        else:
            print(f"Max Invariant set not converged " f"(from class {self.__class__.__name__})")
        return O

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        
        if hasattr(self, 'ocp_eps') and hasattr(self, 'eps_min'):
            # solve the slack optimization first
            self.x0_var_eps.value = x0
            self.ocp_eps.solve(solver=cp.PIQP)
            assert self.ocp_eps.status == cp.OPTIMAL
            self.eps_min.value = self.eps_var.value[:,:-1] # Over the horizon 0 -> N
            
        # solve the controller optimization
        self.x0_var.value = x0
        self.ocp.solve(solver=cp.PIQP)
        assert self.ocp.status == cp.OPTIMAL
        
        x_traj = self.x_var.value
        u_traj = self.u_var.value
        u0 = u_traj[0, 0] 
        #################################################

        return u0, x_traj, u_traj
