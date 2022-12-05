import os
from multiprocessing import Pipe, Process, connection
from functools import partial
import cloudpickle
import pickle
import numpy as np

from MARLLib.envs import ENV, MultiAgentEnv
from MARLLib.utils.buffer import EpisodeBatch
from MARLLib.utils.vec_normalize import VecNormalize, RunningMeanStd
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
    env = wrapper.obj()  # 创建环境
    assert isinstance(env, MultiAgentEnv)
    while True:
        cmd, data = parent.recv()
        if cmd == "step":
            actions = data
            reward, terminated, info = env.step(actions)
            parent.send(
                {
                    "state": env.get_state(),
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
                 test_n_episodes,
                 runner_log_interval,
                 env,
                 env_args):

        self.logger = logger
        self.device = device
        self.batch_size_run = batch_size_run  # 并行 worker 的数量

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

        # IMPROVING: running normalize 是否可以放到 preprocess 中去？
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

        self.steps = 0  # 当前执行总步数
        self.episode_step = None  # 当前在正执行的 episode 中的总步数
        self.log_steps = -1  # -1 表明并未有过记录
        self.test_n_episodes = test_n_episodes  # 测试一次跑多少个 episode
        self.runner_log_interval = runner_log_interval  # 每隔多少步记录一次

        # 信息记录
        self.test_reward = RunningMeanStd(shape=(1,))
        self.train_reward = RunningMeanStd(shape=(1,))
        self.test_stats = {}
        self.train_stats = {}

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

        pre_transition_data = {
            "state": [],
            "obs": []
        }
        for parent in self.parent_conns:
            data = parent.recv()
            # normalize obs and state
            obs_normalized = self.normalizer.normalize_obs(data["obs"], test_mode)
            state_normalized = self.normalizer.normalize_state(data["state"], test_mode)
            pre_transition_data["obs"].append(obs_normalized)
            pre_transition_data["state"].append(state_normalized)

        self.batch.update(pre_transition_data, ts=0)
        self.episode_step = 0

    def rollout(self, test_mode):
        self.reset(test_mode)
        self.controller.init_hidden(self.batch_size_run)

        episode_reward = [0 for _ in range(self.batch_size_run)]
        episode_length = [0 for _ in range(self.batch_size_run)]
        terminated = [False for _ in range(self.batch_size_run)]
        avail_env = [idx for idx, val in enumerate(terminated) if not val]
        episode_final_info = []
        # rollout
        while True:
            actions, old_log_prob = self.controller.forward(self.batch,
                                                            self.episode_step,
                                                            avail_env,
                                                            deterministic=False)
            # actions 增加的维度在 episode_step 上
            # QUESTION: mark_filled 有什么用处？
            # ANSWER: terminated 用来表明环境因失败终止
            #         1. 若存在 terminated[t] == True, 那 t 就是最后一步且走完第 t 步环境失败，最后一步的 Q(t+1) 不需要计算
            #         2. 若不存在 terminated[t] == True, 超时或是成功，最后一步的 Q(t+1) 需要计算
            #         mark_filled 可以不要，使用 mark_filled 可以方便计算
            self.batch.update({"actions": actions}, avail_env, self.episode_step, mark_filled=False)
            self.batch.update({"old_log_prob": old_log_prob}, avail_env, self.episode_step, mark_filled=False)
            last_avail_env = avail_env.copy()

            # 更新下一步的 avail_env
            # 在这里更新和判断跳出的原因是：即使上一步走完 terminated 了，也可能有必要记录这一步的 action（由超时导致的终止）
            avail_env = [idx for idx, val in enumerate(terminated) if not val]
            if len(avail_env) == 0:
                break
            # 将 actions 下达给每个未 terminated 的子环境
            cpu_actions = actions.detach().to("cpu").numpy()
            # 过滤掉失效的 actions
            for avail_env_idx in avail_env:
                idx = last_avail_env.index(avail_env_idx)
                last_avail_env[idx] = -1
            lapsed_env_idx = [idx for idx, val in enumerate(last_avail_env) if val != -1]
            cpu_actions = np.delete(cpu_actions, lapsed_env_idx, axis=0)

            action_idx = 0
            for idx in avail_env:
                self.parent_conns[idx].send(("step", cpu_actions[action_idx]))
                action_idx += 1

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "obs": []
            }
            # 接收子进程返回的信息
            for idx in avail_env:
                data = self.parent_conns[idx].recv()
                # 更新当前步的 post 信息
                post_transition_data["reward"].append((data["reward"],))
                # QUESTION: 对于失败停止和超步停止，需要进行区分吗？
                # ANSWER: 失败停止，最后一步 t 在计算 td_lambda 的时候是用不上 Q(t+1) 的，超步停止则需要。
                #         这里对超步的判断还是依赖于环境(dual_arm_env)所提供的信息的
                post_transition_data["terminated"].append((data["terminated"] and not data["info"]["timeout"],))
                # 更新 episode_final_info，环境顺序无所谓
                if data["terminated"]:
                    episode_final_info.append(data["info"])

                # 更新下一步的 pre 信息
                pre_transition_data["obs"].append(self.normalizer.normalize_obs(data["obs"], test_mode))
                pre_transition_data["state"].append(self.normalizer.normalize_state(data["state"], test_mode))

                episode_reward[idx] += data["reward"]
                episode_length[idx] += 1
                terminated[idx] = data["terminated"]
                if not test_mode:
                    # test 模式下步数不计入
                    self.steps += 1

            self.batch.update(post_transition_data, avail_env, self.episode_step, mark_filled=False)
            self.episode_step += 1
            self.batch.update(pre_transition_data, avail_env, self.episode_step, mark_filled=True)

        # 记录信息
        if test_mode:
            for item in episode_final_info:
                for k, v in item.items():
                    self.test_stats.update({k: self.test_stats.get(k, 0)+v})
            self.test_stats.update({"n_episodes": self.test_stats.get("n_episodes", 0)+self.batch_size_run})
            self.test_stats.update({"steps": sum(episode_length)+self.test_stats.get("steps", 0)})
            self.test_reward.update(np.mean(episode_reward), np.var(episode_reward), self.batch_size_run)
        else:
            for item in episode_final_info:
                for k, v in item.items():
                    self.train_stats.update({k: self.train_stats.get(k, 0)+v})
            self.train_stats.update({"n_episodes": self.train_stats.get("n_episodes", 0)+self.batch_size_run})
            self.train_stats.update({"steps": sum(episode_length)+self.train_stats.get("steps", 0)})
            self.train_reward.update(np.mean(episode_reward), np.var(episode_reward), self.batch_size_run)

        # 打印日志
        if test_mode and (self.test_stats.get("n_episodes", 0) == self.test_n_episodes):
            self.logger.log_stat("test_reward_mean", self.test_reward.mean, self.steps)
            self.logger.log_stat("test_reward_std", self.test_reward.var, self.steps)
            for k, v in self.test_stats.items():
                if k is not "n_episodes":
                    if k is "steps":
                        self.logger.log_stat("test_" + k + "_average", v / self.test_stats["n_episodes"], self.steps)
                    else:
                        self.logger.log_stat("test_" + k + "_rate", v/self.test_stats["n_episodes"], self.steps)
            # 清空记录
            self.test_stats.clear()
            self.test_reward.clear()

        if not test_mode and (self.steps - self.log_steps >= self.runner_log_interval):
            self.logger.log_stat("reward_mean", self.train_reward.mean, self.steps)
            self.logger.log_stat("reward_std", self.train_reward.var, self.steps)
            for k, v in self.train_stats.items():
                if k is not "n_episodes":
                    if k is "steps":
                        self.logger.log_stat("train_" + k + "_average", v/self.train_stats["n_episodes"], self.steps)
                    else:
                        self.logger.log_stat("train_" + k + "_rate", v/self.train_stats["n_episodes"], self.steps)
            # 清空记录
            self.train_stats.clear()
            self.train_reward.clear()
            self.log_steps = self.steps

        return self.batch

    def get_env_info(self):
        return self.env_info

    def close_env(self):
        for parent in self.parent_conns:
            parent.send(("close", None))

    def save_normalizer(self, path):
        self.normalizer.save(os.path.join(path, "vec_normalize.pkl"))

    def load_normalizer(self, path):
        self.normalizer = VecNormalize.load(os.path.join(path, "vec_normalize.pkl"))
