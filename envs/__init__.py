from functools import partial

from .multi_agent_env import MultiAgentEnv
from .water_world import WaterWorld


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


ENV = {"waterworld": partial(env_fn, env=WaterWorld)}
