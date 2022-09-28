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

    def __init__(self, epsilon_start, epsilon_finish, epsilon_anneal_time, test_greedy):
        self.epsilon = None
        self.epsilon_schedule = DecayThenFlatSchedule(epsilon_start, epsilon_finish, epsilon_anneal_time, "linear")
        # test 参数
        self.test_greedy = test_greedy

    def select_actions(self, controller_outputs, avail_actions, t_env, test_mode=False):
        # 将不可行的 action 剔除
        feasible_actions = controller_outputs.clone()
        feasible_actions[avail_actions == 0.0] = 0.0
        # 得到 epsilon
        self.epsilon = self.epsilon_schedule.eval(t_env)

        if test_mode and self.test_greedy:
            # greedy
            picked_actions = feasible_actions.max(dim=2)[1]
        else:
            # epsilon-greedy
            picked_actions = Categorical(feasible_actions).sample()
            random_actions = Categorical(avail_actions).sample()
            # picked_actions 中每个 action 被选择的概率为 1 - epsilon
            # TODO: 这样是不是打破了动作之间的关联性？
            random_numbers = th.rand_like(controller_outputs[:, :, 0])
            select_random_factor = random_numbers < self.epsilon

            picked_actions = select_random_factor * random_actions + (1-select_random_factor)*picked_actions

        # 确保选择的 action 都是可行的
        # unsqueeze 的作用是在扩展指定位置的维度 -- 1
        # squeeze 的作用是把 tensor 中大小为 1 的维度
        # gather 把指定下标的数据从源数据中提取出来
        assert (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all()

        return picked_actions


SELECTOR = {"multinomial": MultinomialActionSelector}
