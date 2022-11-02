from functools import partial

from .multi_agent_env import MultiAgentEnv
from .dual_arm_env import DualArmEnv
from .dual_arm_env_continuous import DualArmContinuousEnv
from .dual_arm_rod_env_continuous import DualArmRodContinuousEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


ENV = {"dualarm": partial(env_fn, env=DualArmEnv),
       "dualarmcontinuous": partial(env_fn, env=DualArmContinuousEnv),
       "dualarmrodcontinuous": partial(env_fn, env=DualArmRodContinuousEnv)}
