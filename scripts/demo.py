import os
import sys
import time
from itertools import cycle

import gymnasium as gym
import numpy as np

sys.path.append(os.getcwd())
import env.register_env
from env.adversaries import Chaser, RandomMover, TrajectoryFollower


def make_adversaries(window, dt, max_accel, max_speed):
    return [RandomMover(i, window, dt, max_accel, max_speed) for i in range(3)]


def run_demo(episodes=3):
    window, dt, max_speed, max_accel = 5.0, 0.1, 1, 0.5
    adversaries = make_adversaries(window, dt, max_accel, max_speed)

    env = gym.make(
        "ReachAvoid-v0",
        adversaries=adversaries,
        window=window,
        dt=dt,
        max_speed=max_speed,
        max_accel=max_accel,
    )

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        print(f"Episode {ep+1}")
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            env.render()
            time.sleep(dt)
            steps += 1

        print(f" Finished in {steps} steps, reward={reward}\n")

    env.close()


if __name__ == "__main__":
    run_demo()
