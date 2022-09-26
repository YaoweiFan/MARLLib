import os
from os.path import dirname, abspath
import yaml
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
import torch as th
import datetime

from utils.logging import get_logger
from utils.function import deep_copy
from run import run

# 实验创建
ex = Experiment("MARL")
# 日志输出
logger = get_logger()
ex.logger = logger
# stdout/stderr 输出至文件  
SETTINGS['CAPTURE_MODE'] = "fd"
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def my_main(_run, _config, _log):
    config = deep_copy(_config)
    # 为不同模块设置 seed
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]
    # 实验 ID
    config["unique_token"] = "{}__{}".format(config["name"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # 框架入口
    run(_run, config, _log)


if __name__ == '__main__':
    # 载入算法参数
    with open(os.path.join(os.path.dirname(__file__), "config", "offpg.yaml"), "r") as f:
        config_dict = yaml.load(f, yaml.FullLoader)
    # 载入环境参数
    env_name = config_dict["env"]
    with open(os.path.join(os.path.dirname(__file__), "config", env_name+".yaml"), "r") as f:
        config_dict["env_args"] = yaml.load(f, yaml.FullLoader)
    # 载入测试参数
    # with open(os.path.join(os.path.dirname(__file__), "config", "test.yaml"), "r") as f:
    #     test_config = yaml.load(f)

    # 把所有参数放入 sacred
    ex.add_config(config_dict)
    # 设置 sacred 的数据存储路径
    save_path = os.path.join(dirname(abspath(__file__)), "results/sacred")
    logger.info("Saving to FileStorageObserver in results/sacred.")
    ex.observers.append(FileStorageObserver(save_path))

    ex.run_commandline()
