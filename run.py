import os
from os.path import dirname, abspath
import numpy as np
import torch as th
from types import SimpleNamespace
import pprint

from utils.logging import Logger
from runners.parallel_runner import ParallelRunner


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
    runner = ParallelRunner(args=args, logger=logger)



def run(run, config, log):
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
    logger.setup_sacred(run)

    # 启动训练
    run_sequential(args=args, logger=logger)