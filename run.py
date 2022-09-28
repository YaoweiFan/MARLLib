import os
import time
from os.path import dirname, abspath
import numpy as np
import torch as th
from types import SimpleNamespace
import pprint

from runners.parallel_runner import ParallelRunner
from agents.controller import Controller
from utils.logging import Logger
from utils.buffer import ReplayBuffer
from utils.preprocess import OneHot


def config_sanity_check_and_adjust(config, log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")
    config["device"] = "cuda" if config["use_cuda"] else "cpu"

    # test_nepisode 应该是 batch_size_run 的整数倍
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]


def run_sequential(args, logger):
    runner = ParallelRunner(logger, args.device, args.batch_size_run, args.env, args.env_args)
    env_info = runner.get_env_info()

    # 创建 on_policy buffer 和 off_policy buffer
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents # 如果是每个 agent 都拥有一份，shape 就应该增加一维
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]) # 对 actions 需要进行预处理
    }
    # 创建 controller
    controller = Controller(scheme, args.batch_size_run, args.n_agents, args.agent_output_type,
                       args.obs_last_action, args.obs_agent_id, args.mask_before_softmax,
                       args.rnn_hidden_dim, args.n_actions, args.action_selector, args.epsilon_start,
                       args.epsilon_finish, args.epsilon_anneal_time, args.test_greedy)

    runner.setup(scheme, groups, preprocess, controller)

    # 创建 learner
    learner =

    on_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"]+1,
                             preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)
    off_buffer = ReplayBuffer(scheme, groups, args.off_buffer_size, env_info["episode_limit"]+1,
                              preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)

    # 从最近的 checkpoint 恢复
    if args.checkpoint_path != "":
        # TODO：载入......
        pass

    # 开始训练
    episode = 0
    last_test_timestep = 0
    last_log_timestep = 0
    model_save_time = 0

    start_time = time.time()

    logger.info("Beginning training for {} timesteps ...".format(args.t_max))
    while runner.steps <= args.t_max:
        # rollout， 每个子进程走完一个 episode
        episode_batch = runner.run(test_mode=False)
        on_buffer.insert_episode_batch(episode_batch)
        off_buffer.insert_episode_batch(episode_batch)



def run(run_sacred, config, log):
    # 配置参数的检查和调整
    config_sanity_check_and_adjust(config, log)
    args = SimpleNamespace(**config)

    logger = Logger(log)
    logger.info("Experiment Parameters:")
    experiment_params = pprint.pformat(config, indent=4, width=1)
    logger.info(experiment_params)

    # 设置 tensorboard
    if config["use_tensorboard"]:
        tb_log_dir = os.path.join(dirname(abspath(__file__)), "results/tb_logs/" + config["unique_token"])
        logger.setup_tb(tb_log_dir)
    # 开启 sacred
    logger.setup_sacred(run_sacred)

    # 启动训练
    run_sequential(args=args, logger=logger)