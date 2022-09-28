from functools import partial

from multi_agent_env import MultiAgentEnv
from dual_arm_env import DualArmEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


ENV = {"dualarm": partial(env_fn, env=DualArmEnv)}
