import numpy as np
import casadi as ca
from typing import Tuple
from control import dlqr

class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """
    # ================================================================================
    #                           INIT
    # ================================================================================
    def __init__(self, rocket, H, xs, us):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """        
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x,u: rocket.f_symbolic(x,u)[0]
        
        # parameters
        self.H = H
        self.xs = xs
        self.us = us
        self.dt = rocket.Ts                 # use the same Ts as the simulator
        self.N  = int(np.round(H / self.dt))  # horizon steps

        self.A_lin, self.B_lin = rocket.linearize(xs, us)
        self._setup_controller()

    # ================================================================================
    #                           SETUP CONTROLLER
    # ================================================================================
    def _setup_controller(self) -> None: 
        """
        self.f = continuos time function
        self.f_d = discretized function by rk4
        """
        Q = np.diag([1, 1, 1,    # omega
                     200, 200, 10000, # phi
                     100, 100, 300,        # vitesses lin
                     1000, 1000, 1000]) # positions xyz
        R = np.diag([3000, 3000, 1, 0.1])

        _, Qf, _ = dlqr(self.A_lin, self.B_lin, Q, R)

        self.opti = ca.Opti()

        # decision (optimisation) variables
        self.X = self.opti.variable(12, self.N + 1)
        self.U = self.opti.variable(4, self.N)
        self.X0 = self.opti.parameter(12, 1)

        self.roll = self.X[5, :]
        self.px = self.X[9, :]
        self.py = self.X[10, :]
        self.pz = self.X[11, :]

        # cost function
        cost = 0
        for k in range(self.N):
            dx = self.X[:, k] - self.xs
            du = self.U[:, k] - self.us
            cost += dx.T @ Q @ dx + du.T @ R @ du

        dxN = self.X[:, self.N] - self.xs
        cost += dxN.T @ Qf @ dxN

        self.opti.minimize(cost)

        # dynamics constraints
        for k in range(self.N):
            self.opti.subject_to(self.X[:, k + 1] == self.f_d(self.X[:, k], self.U[:, k]))
            
        # initial conditions
        self.opti.subject_to(self.X[:, 0] == self.X0)
        
        # constraint on Beta
        self.opti.subject_to((-np.deg2rad(80) <= self.X[4,:]) <= np.deg2rad(80))
    
        # constraint on Z
        self.opti.subject_to(self.X[-1,:] >= 0)
        
        # constraint on U
        self.opti.subject_to((-np.deg2rad(15) <= self.U[0,:]) <= np.deg2rad(15))   # delat1
        self.opti.subject_to((-np.deg2rad(15) <= self.U[1,:]) <= np.deg2rad(15))  # delata2
        self.opti.subject_to((40 <= self.U[2,:]) <= 80)  # P_avg
        self.opti.subject_to((-20 <= self.U[3,:]) <= 20)  # P_diff

        # set solver
        options = {
            "print_time": False,
            "ipopt": {"sb": "yes", "print_level": 0, "tol": 1e-3},
            "expand": True
        }
        self.opti.solver("ipopt", options)

    # ================================================================================
    #                           GET U
    # ================================================================================
    def get_u(self, t0: float, x0: np.ndarray):
        self.opti.set_value(self.X0, x0)
        sol = self.opti.solve()
        u_ol = sol.value(self.U)            # (4, N)
        x_ol = sol.value(self.X)            # (12, N+1)
        u0 = u_ol[:, 0]                     # (4,)
        t_ol = t0 + np.arange(self.N + 1) * self.dt   # (N+1,)
        return u0, x_ol, u_ol, t_ol

    
    # ================================================================================
    #                           Helper Functions
    # ================================================================================
    def f_d(self, x, u): 
        """Discrete time dynamics using RK4 integration."""
        k1 = self.dt * self.f(x, u)
        k2 = self.dt * self.f(x + k1 / 2, u)
        k3 = self.dt * self.f(x + k2 / 2, u)
        k4 = self.dt * self.f(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
