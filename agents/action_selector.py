import torch as th
import numpy as np
from torch.distributions import Categorical


class DecayThenFlatSchedule:
    """衰减（线性或指数）-->平稳"""

    def __init__(self, start, finish, time_length, decay):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, t):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * t)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- t / self.exp_scaling)))


class MultinomialActionSelector:

    def __init__(self, epsilon_start, epsilon_finish, epsilon_anneal_time):
        self.epsilon_schedule = DecayThenFlatSchedule(epsilon_start, epsilon_finish, epsilon_anneal_time, "linear")
        self.epsilon = self.epsilon_schedule.eval(0)

    def select_actions(self, controller_outputs, avail_actions, t_env, greedy):
        # 将不可行的 action 剔除
        feasible_actions = controller_outputs.clone()
        feasible_actions[avail_actions == 0.0] = 0.0
        # 得到 epsilon
        self.epsilon = self.epsilon_schedule.eval(t_env)

        if greedy:
            # greedy policy
            picked_actions = feasible_actions.max(dim=2)[1]
        else:
            # epsilon-greedy policy
            picked_actions = Categorical(feasible_actions).sample()

        # 确保选择的 action 都是可行的
        # unsqueeze 的作用是在扩展指定位置的维度 -- 1
        # squeeze 的作用是把 tensor 中大小为 1 的维度
        # gather 把指定下标的数据从源数据中提取出来
        assert (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all()
        return picked_actions

    def get_epsilon(self):
        return self.epsilon
