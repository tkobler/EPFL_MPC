from unicodedata import name
import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt


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

    def _setup_controller(self) -> None: 
        #################################################
        # YOUR CODE HERE

        self.ocp = ...

        # YOUR CODE HERE
        #################################################
        
        
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
                ((1,2), r'$(\theta,\ v)$'),
                ((0,2), r'$(\omega,\ v)$')]

                fig, axs = plt.subplots(1, 3, figsize=(15, 4))  # exactly 3 panels
                fig.suptitle(f"Invariant set projections, {self.__class__.__name__}")

                for ax, (dims, label) in zip(axs, pairs):
                    O.projection(dims=dims).plot(ax)
                    ax.set_title(label)
                    ax.grid(True)

                plt.tight_layout()
                plt.show()
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title(rf"3D projection $(\omega,\theta,v)$, {self.__class__.__name__}")

                O.projection(dims=(0,1,2)).plot(ax)  # only works if your Polyhedron supports 3D plotting
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
            print(f"Not converged... " f"(from class {self.__class__.__name__})")
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
        # YOUR CODE HERE
        x0_sub = x0
        self.x0_var.value = x0_sub
        self.ocp.solve(solver=cp.PIQP)
        assert self.ocp.status == cp.OPTIMAL
        
        #! These are not the delta values, these are already in global frame
        x_traj = self.x_var.value
        u_traj = self.u_var.value
        u0 = u_traj[0, 0] 
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
