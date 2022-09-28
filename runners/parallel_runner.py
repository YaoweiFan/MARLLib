from multiprocessing import Pipe, Process, connection
from functools import partial
import cloudpickle
import pickle

from MARLLib.envs import ENV, MultiAgentEnv
from MARLLib.utils.buffer import EpisodeBatch
from MARLLib.utils.vec_normalize import VecNormalize
from MARLLib.agents.controller import Controller


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, obj):
        self.obj = obj
    def __getstate__(self):
        return cloudpickle.dumps(self.obj)
    def __setstate__(self, obj):
        self.obj = pickle.loads(obj)


def env_worker(parent: connection.Connection, wrapper: CloudpickleWrapper):
    env = wrapper.obj()
    assert isinstance(env, MultiAgentEnv)
    while True:
        cmd, data = parent.recv()
        if cmd == "step":
            actions = data
            reward, terminated, info = env.step(actions)
            parent.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
                    "reward": reward,
                    "terminated": terminated,
                    "info": info
                }
            )
            continue
        if cmd == "reset":
            env.reset()
            parent.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs()
                }
            )
            continue
        if cmd == "close":
            env.close()
            parent.close()
            break
        if cmd == "get_stats":
            parent.send(env.get_stats())
            continue
        if cmd == "get_env_info":
            parent.send(env.get_env_info())
            continue

        raise NotImplementedError


class ParallelRunner:

    def __init__(self,
                 logger,
                 device,
                 batch_size_run,
                 clip_obs,
                 clip_state,
                 epsilon,
                 use_running_normalize,
                 env,
                 env_args):

        self.logger = logger
        self.device = device
        self.batch_size_run = batch_size_run # 并行 worker 的数量

        # 创建通信管道
        self.parent_conns, worker_conns = zip(*[Pipe() for _ in range(self.batch_size_run)])
        # 创建子进程
        env_fn = partial(ENV[env], **env_args)
        self.processes = [Process(target=env_worker,
                           args=(worker_conn, CloudpickleWrapper(env_fn))) for worker_conn in worker_conns]
        # 开启子进程
        for process in self.processes:
            # 主进程结束，子进程会强制结束
            process.daemon = True
            process.start()

        # 获取环境信息
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.normalizer = VecNormalize(self.env_info["obs_shape"]*self.env_info["n_agents"],
                                       self.env_info["state_shape"],
                                       clip_obs,
                                       clip_state,
                                       epsilon,
                                       use_running_normalize)
        self.scheme = None
        self.groups = None
        self.preprocess = None
        self.controller = None
        self.batch = None

        self.steps = 0 # 当前执行总步数
        self.episode_step = None # 当前在正执行的 episode 中的总步数


    def setup(self, scheme, groups, preprocess, controller: Controller):
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.controller = controller

    def reset(self, test_mode):
        # 创建一个新的 batch
        self.batch = EpisodeBatch(self.scheme, self.groups, self.batch_size_run, self.episode_limit+1,
                                  preprocess=self.preprocess, device=self.device)
        # reset 子线程环境
        for parent in self.parent_conns:
            parent.send(("reset", None))

        pre_transition_data = { "state": [], "avail_actions": [], "obs": [] }
        for parent in self.parent_conns:
            data = parent.recv()
            # normalize obs and state
            obs_normalized = self.normalizer.normalize_obs(data["obs"], test_mode)
            state_normalized = self.normalizer.normalize_state(data["state"], test_mode)
            pre_transition_data["obs"].append(obs_normalized)
            pre_transition_data["state"].append(state_normalized)
            pre_transition_data["avail_actions"].append(data["avail_actions"])

        self.batch.update(pre_transition_data, ts=0)

        self.episode_step = 0

    def run(self, test_mode):
        self.reset(test_mode)

        while True:
            actions = self.controller.select_actions(self.batch, self.episode_step, self.steps, envs_not_terminated, test_mode)

    def collect_one_epsoide(self):


    def get_env_info(self):
        return self.env_info
