import numpy as np
import cvxpy as cp


class MPCSolver:
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int,
        pos_bound: float,
        vel_bound: float,
        acc_bound: float,
        dt: float = 1.0,
        verbose: bool = False,
    ):
        """
        A, B         : system matrices (n×n, n×d)
        Q, R         : cost weights on state and control (n×n, d×d)
        horizon      : number of control steps (N)
        pos_bound    : max |px|, |py|
        vel_bound    : max |vx|, |vy|
        acc_bound    : max |ax|, |ay|
        dt           : timestep (only used if you rebuild A,B inside)
        solver       : CVXPY solver
        verbose      : solver verbosity
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = horizon
        self.dt = dt

        # dimensions
        self.n = A.shape[0]
        self.d = B.shape[1]

        # Position:  0 <= px,py <= pos_bound
        posA = np.array(
            [
                [1, 0, 0, 0],  #  px <= pos_bound
                [-1, 0, 0, 0],  # -px <= 0   ⇒  px >= 0
                [0, 1, 0, 0],  #  py <= pos_bound
                [0, -1, 0, 0],  # -py <= 0   ⇒  py >= 0
            ]
        )
        posb = np.array([pos_bound, 0, pos_bound, 0])

        # Velocity: -vel_bound <= vx,vy <= vel_bound
        velA = np.array(
            [
                [0, 0, 1, 0],  #  vx <= vel_bound
                [0, 0, -1, 0],  # -vx <= vel_bound
                [0, 0, 0, 1],  #  vy <= vel_bound
                [0, 0, 0, -1],  # -vy <= vel_bound
            ]
        )
        velb = np.array([vel_bound] * 4)

        # stack them
        self.Fx = np.vstack([posA, velA])  # shape (8×4)
        self.bx = np.hstack([posb, velb])  # length 8

        # controls remain symmetric ±acc_bound
        self.Fu = np.vstack([np.eye(self.d), -np.eye(self.d)])
        self.bu = np.array([acc_bound] * self.d * 2)

        self.verbose = verbose

    def solve(
        self,
        x0: np.ndarray,
        x_goal: np.ndarray,
        obs_preds: np.ndarray,
        obs_radii: np.ndarray,
    ):
        """
        Solve the MPC with obstacle half‐space approximations.

        x0        : (n,)      initial state
        x_goal    : (n,)      goal state
        obs_preds : (m, N+1, 2)  predicted obstacle centers for m obstacles over the horizon
        obs_radii : (m, N+1)     radius of each obstacle at each timestep

        returns (x_traj, u_seq, obj_val)
        """
        N, n, d = self.N, self.n, self.d
        m = obs_preds.shape[0]  # number of obstacles

        # decision variables
        x = cp.Variable((n, N + 1))
        u = cp.Variable((d, N))

        cost = 0
        constraints = []

        # initial condition
        constraints += [x[:, 0] == x0]

        # dynamics, bounds, cost
        for k in range(N):
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]
            constraints += [
                self.Fx @ x[:, k] <= self.bx,
                self.Fu @ u[:, k] <= self.bu,
            ]
            cost += cp.quad_form(x[:, k] - x_goal, self.Q)
            cost += cp.quad_form(u[:, k], self.R)

        # terminal state bound
        constraints += [self.Fx @ x[:, N] <= self.bx]
        # final cost (optional)
        cost += cp.quad_form(x[:, N] - x_goal, self.Q)

        # obstacle avoidance via half‐space approximation
        for i in range(m):
            for k in range(N + 1):
                center = obs_preds[i, k]  # shape (2,)
                radius = obs_radii[i, k]  # scalar

                # compute a fixed normal at time k:
                dir_vec = x0[:2] - center
                if np.linalg.norm(dir_vec) < 1e-6:
                    # fallback if state exactly at center
                    normal = np.array([1.0, 0.0])
                else:
                    normal = dir_vec / np.linalg.norm(dir_vec)

                # linear constraint: normal · (pos_k - center) >= radius
                constraints += [normal @ (x[:2, k] - center) >= radius]

        # build & solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=self.verbose, warm_start=True)

        return x.value, u.value, prob.value

    def solve_fallback(
        self,
        x0: np.ndarray,
        obs_preds: np.ndarray,
        obs_radii: np.ndarray,
    ):
        """
        Phase I fallback: keep all original Fx/Fu bounds hard, but introduce
        slack only for obstacle half‐spaces.  Minimizes sum of s_obs.

        Returns: x_fallback, u_fallback, total_obstacle_violation
        """
        N, n, d = self.N, self.n, self.d
        m = obs_preds.shape[0]

        # decision vars
        x = cp.Variable((n, N + 1))
        u = cp.Variable((d, N))
        s_obs = cp.Variable((m, N + 1), nonneg=True)

        constraints = []
        constraints += [x[:, 0] == x0]

        for k in range(N):
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]
            constraints += [
                self.Fx @ x[:, k] <= self.bx,
                self.Fu @ u[:, k] <= self.bu,
            ]
        constraints += [self.Fx @ x[:, N] <= self.bx]

        for i in range(m):
            for k in range(N + 1):
                center = obs_preds[i, k]
                radius = obs_radii[i, k]

                # pick a fixed normal direction
                dir_vec = x0[:2] - center
                if np.linalg.norm(dir_vec) < 1e-6:
                    normal = np.array([1.0, 0.0])
                else:
                    normal = dir_vec / np.linalg.norm(dir_vec)

                # now:      normal·(x[:2,k]-center) + s_obs[i,k] ≥ radius
                constraints += [normal @ (x[:2, k] - center) + s_obs[i, k] >= radius]

        # objective: minimize obstacle‐slack
        obj = cp.sum(s_obs)

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(
            solver=cp.SCS,
            verbose=self.verbose,
            warm_start=True,
        )

        x_fb = x.value
        u_fb = u.value
        total_slack = obj.value

        return x_fb, u_fb, total_slack


class MIMPCSolver(MPCSolver):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int,
        pos_bound: float,
        vel_bound: float,
        acc_bound: float,
        dt: float = 1.0,
    ):
        super().__init__(A, B, Q, R, horizon, pos_bound, vel_bound, acc_bound, dt)
        self.M = 1e5
        self.eps = 1e-3
        self.solver = cp.SCIP

    def solve(
        self,
        x0: np.ndarray,
        x_goal: np.ndarray,
        obs_preds: np.ndarray,
        obs_radii: np.ndarray,
    ):
        """
        Solve the MIMPC with obstacle half‐space approximations.

        x0        : (n,)      initial state
        x_goal    : (n,)      goal state
        obs_preds : (m, N+1, 2)  predicted obstacle centers for m obstacles over the horizon
        obs_radii : (m, N+1)     radius of each obstacle at each timestep

        returns (x_traj, u_seq, obj_val)
        """
        N, n, d = self.N, self.n, self.d
        m = obs_preds.shape[0]  # number of obstacles

        # decision variables
        x = cp.Variable((n, N + 1))
        u = cp.Variable((d, N))
        b = [[cp.Variable(4, boolean=True) for k in range(N + 1)] for i in range(m)]

        cost = 0
        constraints = []

        # initial condition
        constraints += [x[:, 0] == x0]

        # dynamics, bounds, cost
        for k in range(N):
            constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]
            constraints += [
                self.Fx @ x[:, k] <= self.bx,
                self.Fu @ u[:, k] <= self.bu,
            ]
            cost += cp.quad_form(x[:, k] - x_goal, self.Q)
            cost += cp.quad_form(u[:, k], self.R)

        # terminal state bound
        constraints += [self.Fx @ x[:, N] <= self.bx]
        # final cost (optional)
        cost += cp.quad_form(x[:, N] - x_goal, self.Q)

        # obstacle avoidance via half‐space approximation
        for i in range(m):
            for k in range(N + 1):
                cx, cy = obs_preds[i, k]
                r = obs_radii[i, k]
                xmin, xmax = cx - r, cx + r
                ymin, ymax = cy - r, cy + r
                bs = b[i][k]
                constraints += [
                    x[0, k] <= xmin - self.eps + self.M * (1 - bs[0]),
                    x[0, k] >= xmax + self.eps - self.M * (1 - bs[1]),
                    x[1, k] <= ymin - self.eps + self.M * (1 - bs[2]),
                    x[1, k] >= ymax + self.eps - self.M * (1 - bs[3]),
                    cp.sum(bs) >= 1,
                ]

        # build & solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(
            solver=self.solver,
            verbose=self.verbose,
            warm_start=True,
            scip_params={
                "limits/time": 4.0,
                "limits/gap": 0.02,
            },
        )

        return x.value, u.value, prob.value
