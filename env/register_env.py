# register_env.py
from gymnasium.envs.registration import register

register(
    id="ReachAvoid-v0",
    entry_point="env.reach_avoid_env:ReachAvoidEnv",
    max_episode_steps=200,
    kwargs={
        "window": 5.0,
        "dt": 0.1,
        "max_speed": 1.0,
        "max_accel": 1.0,
        "adversaries": None
    },
)
