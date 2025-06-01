from typing import Iterable

import numpy as np


class Adversary:
    def __init__(
        self, idx: int, window: float, dt: float, max_accel: float, max_speed: float
    ):

        self.idx = idx
        self.dt = dt
        self.window = window
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.center = np.array([window / 2, window / 2])

    def step(self, states):
        raise NotImplementedError


class RandomMover(Adversary):
    def __init__(
        self, idx: int, window: float, dt: float, max_accel: float, max_speed: float
    ):

        super().__init__(idx, window, dt, max_accel, max_speed)

        self.A = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

    def step(self, states):
        own_state = states["adversaries"][self.idx]
        pos = own_state[:2]
        a = np.random.uniform(-self.max_accel, self.max_accel, size=2)

        margin = 0.2 * self.window
        if (pos < margin).any() or (pos > self.window - margin).any():
            to_center = self.center - pos
            if np.linalg.norm(to_center) > 1:
                to_center = to_center / np.linalg.norm(to_center) * self.max_accel
                a = 0.5 * a + 0.5 * to_center

        return self.A @ own_state + self.B @ a


class Chaser(Adversary):
    def __init__(
        self, idx: int, window: float, dt: float, max_accel: float, max_speed: float
    ):

        super().__init__(idx, window, dt, max_accel, max_speed)

        self.A = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

    def step(self, states):
        own_state = states["adversaries"][self.idx]
        agent_pos = states["agent"][:2]
        adv_pos, adv_vel = own_state[:2], own_state[2:]

        delta = agent_pos - adv_pos
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.zeros(4)

        v_target = delta / dist * self.max_speed
        a = np.clip((v_target - adv_vel), -self.max_accel, self.max_accel)
        return self.A @ own_state + self.B @ a


class TrajectoryFollower(Adversary):
    def __init__(
        self,
        idx: int,
        window: float,
        dt: float,
        max_accel: float,
        max_speed: float,
        waypoints: Iterable,
    ):

        super().__init__(idx, window, dt, max_accel, max_speed)
        self.waypoints = waypoints

    def step(self, states):
        return next(self.waypoints)


class RandomChaser(Adversary):
    """
    A chaser that acts randomly with probability p.
    """

    def __init__(
        self,
        idx: int,
        window: float,
        dt: float,
        max_accel: float,
        max_speed: float,
        p: float,  # probability of acting randomly
    ):
        super().__init__(idx, window, dt, max_accel, max_speed)
        self.p = p

        self.A = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

    def step(self, states):
        own_state = states["adversaries"][self.idx]
        pos, vel = own_state[:2], own_state[2:]

        if np.random.rand() < self.p:
            # RandomMover logic
            a = np.random.uniform(-self.max_accel, self.max_accel, size=2)
            margin = 0.2 * self.window
            if (pos < margin).any() or (pos > self.window - margin).any():
                to_center = self.center - pos
                norm = np.linalg.norm(to_center)
                if norm > 1:
                    to_center = to_center / norm * self.max_accel
                    a = 0.5 * a + 0.5 * to_center
        else:
            # Chaser logic
            agent_pos = states["agent"][:2]
            delta = agent_pos - pos
            dist = np.linalg.norm(delta)
            if dist < 1e-6:
                return np.zeros(4)
            v_target = delta / dist * self.max_speed
            a = np.clip(v_target - vel, -self.max_accel, self.max_accel)

        return self.A @ own_state + self.B @ a
