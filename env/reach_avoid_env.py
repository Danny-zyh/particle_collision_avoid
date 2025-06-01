from typing import List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces  # isort

from env.adversaries import Adversary


class ReachAvoidEnv(gym.Env):
    def __init__(
        self,
        adversaries: List[Adversary],
        window: float = 5.0,
        dt: float = 0.1,
        max_speed: float = 1,
        max_accel: float = 1,
    ):
        super().__init__()
        # TODO: write everything in a config file
        self.dt = dt
        self.window = window  # world is [0,window]×[0,window]
        self.max_accel = max_accel
        self.max_speed = max_speed
        self.num_adversaries = len(adversaries)
        self.adversaries = adversaries
        self.action_space = spaces.Box(
            low=-max_accel, high=max_accel, shape=(2,), dtype=np.float32
        )

        low = np.array(
            [0, 0, -max_speed, -max_speed] * (1 + self.num_adversaries) + [0, 0],
            dtype=np.float32,
        )
        high = np.array(
            [window, window, max_speed, max_speed] * (1 + self.num_adversaries)
            + [window, window],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.states = {
            "agent": np.zeros(4),  # (pos_x, pos_y, vel_x, vel_y)
            "adversaries": [np.zeros(4)] * self.num_adversaries,
            "landmark": np.zeros(4),  # (pos_x, pos_y)
        }

        # A simple double integrator
        # TODO: write dynamic system to config
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

        self.reset()

    def clip_action(self, action):
        return np.clip(action, -self.max_accel, self.max_accel)

    def clip_state(self, state):
        return np.clip(
            state,
            a_min=[0] * 2 + [-self.max_speed] * 2,
            a_max=[self.window] * 2 + [self.max_speed] * 2,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.states["agent"] = np.array(
            [0.5, 0.5, 0, 0]
        )  # TODO: write start state in config

        options = options if options else dict()
        adv_pos = options.get("adv_pos", None)
        if adv_pos is None:
            # random initialization
            adv_pos = np.random.uniform(0, self.window, size=(self.num_adversaries, 2))
        else:
            assert isinstance(adv_pos, np.ndarray) and adv_pos.shape == (
                self.num_adversaries,
                2,
            ), "adv_pos is an ndarray with shape (num_adv, 2)"

        for i in range(self.num_adversaries):
            self.states["adversaries"][i] = np.array(
                [*adv_pos[i], 0, 0]
            )  # TODO: write start state in config

        self.states["landmark"][:2] = np.array(
            [self.window - 0.5, self.window - 0.5]
        )  # TODO: write landmark position to config
        return self.states, {}

    def step(self, action):
        # step agent
        action = self.clip_action(action)
        agent_next_state = self.clip_state(
            self.A @ self.states["agent"] + self.B @ action
        )

        # step adversarial
        adv_next_states = [
            self.clip_state(adv.step(self.states)) for adv in self.adversaries
        ]

        # update state
        self.states["agent"] = agent_next_state
        self.states["adversaries"] = adv_next_states

        # calculate reward and done
        reward = 0
        done = False
        distance_to_goal = np.linalg.norm(agent_next_state - self.states["landmark"])
        if distance_to_goal < 0.2:  # TODO: add success check to config
            done = True
            reward = 1

        for adv in adv_next_states:
            distance_to_adv = np.linalg.norm(agent_next_state[:2] - adv[:2])
            if distance_to_adv < 0.4:  # TODO: add adv check to config
                done = True
                reward = -1
        return self.states, reward, done, False, {}

    def render(self):
        plt.clf()
        plt.scatter(*self.states["agent"][:2], c="blue", s=100, label="You")
        for i in range(self.num_adversaries):
            plt.scatter(
                self.states["adversaries"][i][0],
                self.states["adversaries"][i][1],
                c="red",
                s=100,
                label="Enemy",
            )
        plt.scatter(
            self.states["landmark"][0],
            self.states["landmark"][1],
            c="green",
            s=80,
            marker="*",
            label="Goal",
        )
        plt.xlim(0, self.window)
        plt.ylim(0, self.window)
        plt.legend()
        plt.pause(0.001)

    def close(self):
        pass
