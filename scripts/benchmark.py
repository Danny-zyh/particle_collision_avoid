import os
import numpy as np
import gymnasium as gym
import pickle
import copy
from tqdm import tqdm
from dataclasses import dataclass
import cvxpy as cp
import time

import sys

sys.path.append(os.getcwd())

from conformal_region import ConformalRegion
from env.adversaries import Chaser, RandomMover, TrajectoryFollower, RandomChaser
import env.register_env
from mpc import MPCSolver
from mppi import MPPISolver
from utils import animate_trajectory, animate_with_predictions


@dataclass
class Config:
    # Global
    seed: int = 42
    n_episodes: int = 50

    # Env parameters
    n_chasers: int = 0
    n_random_movers: int = 5
    n_random_chasers: int = 0
    random_chaser_p: float = (
        0.5  # probability of acting randomly, 1 means always random
    )
    window: float = 5.0
    dt: float = 0.1
    max_speed: float = 1.0
    max_accel: float = 3.0

    # Conformal region
    horizon: int = 17
    cp_horizon_max: int = 20
    alpha: float = 0.3
    traj_path: str = "dataset/random_5_p1_1_p5.npy"
    pred_path: str = "dataset/random_5_p1_1_p5_pred_20.npy"

    # Obstacle predictor
    predictor_path: str = "dataset/random_5_p1_1_p5_ridge_model.pkl"
    len_memory: int = 5

    # Solver
    solver_type: str = "mpc"  # or "mppi"


def make_env(config: Config):
    chasers = [
        Chaser(i, config.window, config.dt, config.max_accel, config.max_speed)
        for i in range(config.n_chasers)
    ]

    random_movers = [
        RandomMover(
            i + config.n_chasers,
            config.window,
            config.dt,
            config.max_accel,
            config.max_speed,
        )
        for i in range(config.n_random_movers)
    ]

    random_chasers = [
        RandomChaser(
            i + config.n_chasers + config.n_random_movers,
            config.window,
            config.dt,
            config.max_accel,
            config.max_speed,
            config.random_chaser_p,
        )
        for i in range(config.n_random_chasers)
    ]

    adversaries = chasers + random_movers + random_chasers

    env = gym.make(
        "ReachAvoid-v0",
        adversaries=adversaries,
        window=config.window,
        dt=config.dt,
        max_speed=config.max_speed,
        max_accel=config.max_accel,
    )
    return env


def get_conformal_region(config: Config, n_adversaries: int):
    alpha_agent = 1 - np.power(1 - config.alpha, 1 / n_adversaries)
    print("alpha level for each agent is", alpha_agent)
    return ConformalRegion(
        config.traj_path, config.pred_path, config.cp_horizon_max, alpha_agent
    )


def get_solver(config: Config):
    A = np.array(
        [[1, 0, config.dt, 0], [0, 1, 0, config.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    B = np.array([[0, 0], [0, 0], [config.dt, 0], [0, config.dt]])

    if config.solver_type == "mpc":
        Q = np.eye(4) * 0.01
        R = np.eye(2) * 0.01
        return MPCSolver(
            A,
            B,
            Q,
            R,
            config.horizon,
            config.window,
            config.max_speed,
            config.max_accel,
            config.dt,
        )

    elif config.solver_type == "mppi":
        Q = 1e7 * np.eye(4)
        R = 1e3 * np.eye(2)
        Qf = 1e7 * np.eye(4)
        return MPPISolver(
            A,
            B,
            Q,
            R,
            Qf,
            config.horizon,
            200,
            1.0,
            0.3,
            np.array([-config.max_accel] * 2),
            np.array([config.max_accel] * 2),
            np.array([0, 0, -config.max_speed, -config.max_speed]),
            np.array(
                [config.window, config.window, config.max_speed, config.max_speed]
            ),
            penalty_obs=1e20,
            penalty_state=1e20,
        )
    else:
        raise ValueError(f"Unknown solver type: {config.solver_type}")


def rollout_trajectories(
    init_history,
    len_memory,
    predictor,
    horizon=17,
):
    """
    Autoregressively predict obstacle trajectories.
    Input: (n_adversary, len_memory, 2) obstacle position history
    Output: (n_adversary, horizon+1, 2) obstacle furture state prediction
    """
    m = init_history.shape[0]
    # container: time 0…horizon
    preds = np.zeros((m, horizon + len_memory + 1, 2), dtype=float)
    # fill in the known history
    preds[:, :len_memory, :] = init_history

    for t in range(len_memory, horizon + len_memory + 1):
        # grab the last 5 steps as a flat 10-dim feature
        X_batch = preds[:, t - len_memory : t, :].reshape(m, len_memory * 2)
        # predict next (x,y) for all m obstacles at once
        next_xy = predictor.predict(X_batch)  # → shape (m,2)
        preds[:, t, :] = next_xy

    return preds[
        :,
        len_memory:,
    ]


def policy(
    solver,
    solver_name,
    predictor,
    observation,
    obs_history,
    t,
    CP,
    n_adversaries,
    len_memory,
    horizon,
    log=None,
) -> np.ndarray:
    x_goal = observation["landmark"]
    x0 = observation["agent"]

    # update obstacle memory
    for i, adv in enumerate(obs_history):
        adv.pop(0)
        adv.pop(0)
        adv += list(observation["adversaries"][i][:2])

    obs_preds = rollout_trajectories(
        np.array(obs_history).reshape(n_adversaries, len_memory, 2),
        len_memory,
        predictor,
        horizon,
    )
    obs_radii = np.array(
        [CP.construct_valid_prediction_regions(t, i) for i in range(horizon + 1)]
    )
    obs_radii = np.repeat(obs_radii[None, :], n_adversaries, axis=0) + 0.4

    if log:
        log["obs_preds"].append(obs_preds)
        log["obs_radii"].append(obs_radii)

    if solver_name == "mpc":
        try:
            x, u, prob = solver.solve(x0, x_goal, obs_preds, obs_radii)
        except cp.SolverError:
            x, u, prob = solver.solve_fallback(x0, obs_preds, obs_radii)
        if u is None:
            x, u, prob = solver.solve_fallback(x0, obs_preds, obs_radii)
        return u[:, 0]  # return only the first sequence
    elif solver_name == "mppi":
        u = solver.solve(x0, x_goal, obs_preds, obs_radii)
        if u is None:
            u = np.zeros(2)
        return u
    else:
        raise ValueError(f"Unknown solver type: {solver_name}")


def run_single_episode(config: Config, env, solver, CP, predictor, init_obs):
    n_adversaries = config.n_chasers + config.n_random_movers + config.n_random_chasers
    obs_history = [list(x0[:2]) * config.len_memory for x0 in init_obs["adversaries"]]
    action = env.action_space.sample()
    log = {"obs_preds": [], "obs_radii": []}
    simulation_trajectory = []
    success = False

    for t in tqdm(range(180), desc="Episode Progress"):
        obs, reward, done, _, _ = env.step(action)
        simulation_trajectory.append(copy.deepcopy(obs))

        if done:
            success = reward > 0
            break

        try:
            action = policy(
                solver,
                config.solver_type,
                predictor,
                obs,
                obs_history,
                t,
                CP,
                n_adversaries,
                config.len_memory,
                config.horizon,
                log,
            )
        except Exception as e:
            print(f"Error at time {t}: {e}")
            break

    html = animate_trajectory(simulation_trajectory)

    strings = time.strftime("%Y,%m,%d,%H,%M,%S")
    t = strings.split(",")

    file_name = f"{config.solver_type}_{'_'.join(t)}.html"

    os.makedirs(os.path.join(os.getcwd(), "test"), exist_ok=True)

    with open(os.path.join(os.getcwd(), "test", file_name), "w") as f:
        f.write(html.data)

    return success


def evaluate_solver(config: Config, solver_type: str, init_positions: np.ndarray):
    print(f"\n=== Evaluating {solver_type.upper()} ===")
    config.solver_type = solver_type
    n_adversaries = config.n_chasers + config.n_random_movers + config.n_random_chasers
    env = make_env(config)
    CP = get_conformal_region(config, n_adversaries)
    solver = get_solver(config)

    with open(config.predictor_path, "rb") as f:
        predictor = pickle.load(f)

    successes = 0
    for ep in tqdm(range(config.n_episodes), desc=f"{solver_type.upper()} Episodes"):
        init_obs = env.reset(options={"adv_pos": init_positions[ep]})[0]
        if run_single_episode(config, env, solver, CP, predictor, init_obs):
            successes += 1

    success_rate = successes / config.n_episodes
    print(f"{solver_type.upper()} success rate: {success_rate:.2%}")
    return success_rate


if __name__ == "__main__":
    config = Config()
    config.n_episodes = 10  # you can override this as needed

    # Pre-generate init positions for fair evaluation
    n_adv = config.n_chasers + config.n_random_movers + config.n_random_chasers
    init_positions = np.random.uniform(
        0, config.window, size=(config.n_episodes, n_adv, 2)
    )

    results = {}
    for solver in ["mpc", "mppi"]:
        results[solver] = evaluate_solver(config, solver, init_positions)

    print("\n=== Final Summary ===")
    for solver, rate in results.items():
        print(f"{solver.upper()} success rate: {rate:.2%}")
