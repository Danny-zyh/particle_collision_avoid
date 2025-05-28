import numpy as np

class MPPISolver:
    def __init__(self, A, B, Q, R, Qf, horizon, n_samples, lambda_, sigma, u_min, u_max, x_min, x_max, penalty_obs=1e20, penalty_state=1e20):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.N = horizon
        self.n_samples = n_samples
        self.lambda_ = lambda_
        self.sigma = sigma
        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max
        
        self.penalty_obs = penalty_obs
        self.penalty_state = penalty_state

    def dynamics(self, x, u):
        return self.A @ x + self.B @ u

    def solve(self, x0, x_goal, obs_preds, obs_radii):
        m = obs_preds.shape[0]
        u_nom = np.zeros((self.N, self.B.shape[1]))
        x = x0.copy()

        for _ in range(1):
            noise = self.sigma * np.random.randn(self.n_samples, self.N, self.B.shape[1])
            u_samples = np.clip(u_nom[None, :, :] + noise, self.u_min, self.u_max)
            costs = np.zeros(self.n_samples)

            for i in range(self.n_samples):
                xi = x.copy()
                cost = 0
                for k in range(self.N):
                    ui = u_samples[i, k]
                    xi = self.dynamics(xi, ui)

                    dx = xi - x_goal
                    cost += dx.T @ self.Q @ dx + ui.T @ self.R @ ui

                    for j in range(m):
                        obs_center = obs_preds[j, min(k, obs_preds.shape[1]-1)]
                        obs_radius = obs_radii[j, min(k, obs_radii.shape[1]-1)]
                        if np.linalg.norm(xi[:2] - obs_center) < obs_radius:
                            cost += self.penalty_obs

                    if np.any(xi < self.x_min) or np.any(xi > self.x_max):
                        cost += self.penalty_state

                dx = xi - x_goal
                cost += dx.T @ self.Qf @ dx
                costs[i] = cost

            beta = np.min(costs)
            weights = np.exp(-(costs - beta) / self.lambda_)
            weights /= np.sum(weights)

            du = np.sum(weights[:, None, None] * (u_samples - u_nom[None, :, :]), axis=0)
            u_nom += du

        return u_nom[0]
