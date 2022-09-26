from collections import defaultdict
import logging
import numpy as np
import torch as th
from tensorboard_logger import configure, log_value


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.tb_logger = None
        self.use_sacred = False
        self.sacred_info = None
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        configure(directory_name)
        self.use_tb = True
        self.tb_logger = log_value

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            # item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            item = "{:.4f}".format(self.average_list(k, window))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.info(log_str)

    def average_list(self, k, window):
        array = []
        for stat in self.stats[k][-window:]:
            item = stat[1]
            if th.is_tensor(item):
                item = item.cpu().numpy()

            array.append(item)

        return np.mean(array)

    def info(self, info):
        self.console_logger.info(info)


def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger
