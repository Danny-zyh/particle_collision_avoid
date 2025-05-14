import os
import sys
import time
from tqdm import tqdm
from itertools import cycle

import gymnasium as gym
import numpy as np

sys.path.append(os.getcwd())
import env.register_env
from env.adversaries import Chaser, RandomMover, TrajectoryFollower


def make_adversaries(window, dt, max_accel, max_speed):
    return [
        RandomMover(0, window, dt, max_accel, max_speed),
    ]


def collect_traj(T=200, N=200):
    window, dt, max_speed, max_accel = 5.0, 0.1, 1, 0.5
    dataset = np.zeros((N, T, 2))
    adversaries = make_adversaries(window, dt, max_accel, max_speed)

    # now "ReachAvoid-v0" is registered
    env = gym.make(
        "ReachAvoid-v0",
        adversaries=adversaries,
        window=window,
        dt=dt,
        max_speed=max_speed,
        max_accel=max_accel,
    )

    for ep in tqdm(range(N)):
        obs, _ = env.reset()
        done = False
        steps = 0
        env.reset()
        for t in range(T):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            steps += 1
            dataset[ep, t] = obs["adversaries"][0][:2]

    env.close()
    np.save("dataset/random_5_p1_1_p5", dataset)


if __name__ == "__main__":
    collect_traj()
