from functools import partial

from .multi_agent_env import MultiAgentEnv
from .dual_arm_env import DualArmEnv
from .dual_arm_rod_env import DualArmRodEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


ENV = {"dualarm": partial(env_fn, env=DualArmEnv),
       "dualarmrod": partial(env_fn, env=DualArmRodEnv)}
