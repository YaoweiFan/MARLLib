from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from envs.dualarmenv import DualArmEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["dualarm"] = partial(env_fn, env=DualArmEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "/home/fyw/StarCraftII")

from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])