import os
import torch as th
import numpy as np
import math
from itertools import chain

from MARLLib.utils.distributions import DiagGaussianDistribution
from .fc_agent import FcAgent


class Controller:
    """为多个智能体提供 obs-->action 的映射"""

    def __init__(self,
                 scheme,
                 n_agents,
                 obs_last_action,
                 obs_agent_id,
                 rnn_hidden_dim,
                 action_dim,
                 log_std_init
                 ):

        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_last_action = obs_last_action
        self.obs_agent_id = obs_agent_id
        # 计算输入维度
        input_shape = scheme["obs"]["vshape"]
        if obs_last_action:
            input_shape += scheme["actions"]["vshape"]
        if obs_agent_id:
            input_shape += self.n_agents
        # 创建 agent
        self.agent = FcAgent(input_shape, rnn_hidden_dim, action_dim)
        # 创建 action 采样 distribution
        self.action_distribution = DiagGaussianDistribution(action_dim)
        self.log_std = th.nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=False)

    def _build_inputs(self, ep_batch, episode_step):
        """
               obs: (batch_size_run, n_agents, obs_size)
               last_action: (batch_size_run, n_agents, last_action_size)
               obs_agent_id: (batch_size_run, n_agents, agent_id_size)
        return:
               inputs: (batch_size_run * n_agents, obs_size+last_action_dim+agent_id_size)
        """
        inputs = [ep_batch["obs"][:, episode_step]]
        if self.obs_last_action:
            if episode_step == 0:
                inputs.append(th.zeros_like(ep_batch["actions"][:, episode_step]))
            else:
                inputs.append(ep_batch["actions"][:, episode_step - 1])
        if self.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).expand(ep_batch.batch_size, -1, -1))
        inputs = th.cat([item.reshape(ep_batch.batch_size * self.n_agents, -1) for item in inputs], dim=1)
        return inputs

    def forward(self, ep_batch, episode_step, avail_env=slice(None), deterministic=False):
        # inputs: (batch_size_run * n_agents, obs_size+last_action_dim+agent_id_size)
        inputs = self._build_inputs(ep_batch, episode_step)
        # mean_actions: (batch_size_run * n_agents, action_dim)
        mean_actions = self.agent(inputs)
        distribution = self.action_distribution.proba_distribution(mean_actions, self.log_std)
        # 若 deterministic == True，意味着 action 选择的直接是 mean action
        actions = distribution.get_actions(deterministic=deterministic)
        # 对输出动作作限制
        actions = th.clamp(actions, -1, 1)
        # old_log_prob: (batch_size_run * n_agents, )
        old_log_prob = distribution.log_prob(actions)
        if deterministic:
            return actions.reshape(ep_batch.batch_size, self.n_agents, -1)[avail_env]
        else:
            return actions.reshape(ep_batch.batch_size, self.n_agents, -1)[avail_env], \
                old_log_prob.reshape(ep_batch.batch_size, self.n_agents, -1)[avail_env]

    def evaluate_actions(self, ep_batch, episode_step):
        # inputs: (batch_size_run * n_agents, obs_size+last_action_dim+agent_id_size)
        inputs = self._build_inputs(ep_batch, episode_step)
        # mean_actions: (batch_size_run * n_agents, action_dim)
        mean_actions = self.agent(inputs)
        distribution = self.action_distribution.proba_distribution(mean_actions, self.log_std)
        actions = ep_batch["actions"][:, episode_step].reshape(ep_batch.batch_size*self.n_agents, -1)
        # log_prob: (batch_size_run * n_agents, )
        log_prob = distribution.log_prob(actions)
        return log_prob.reshape(ep_batch.batch_size, self.n_agents, -1)

    def cuda(self):
        self.agent.cuda()
        self.log_std = self.log_std.to("cuda")

    def parameters(self):
        # return chain(self.agent.parameters(), self.distribution_param())
        return self.agent.parameters()

    # def distribution_param(self):
    #     yield self.log_std

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path, record_param=False):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if record_param:
            self.record(os.path.join(path, "parameters"))
            raise Exception("Controller network loaded successfully!")

    def variance_reduce(self):
        if self.log_std[0] > math.log(0.5):
            self.log_std += math.log(0.999998)

    def record(self, path):
        # 记录 agent 网络参数
        np.savetxt(path + 'fc1_weight.txt', self.agent.state_dict()["fc1.weight"].cpu().numpy())
        np.savetxt(path + 'fc1_bias.txt', self.agent.state_dict()["fc1.bias"].cpu().numpy())

        np.savetxt(path + 'gru_wir.txt', self.agent.state_dict()["rnn.weight_ih"].cpu().numpy()[:64, :])
        np.savetxt(path + 'gru_wiz.txt', self.agent.state_dict()["rnn.weight_ih"].cpu().numpy()[64:128, :])
        np.savetxt(path + 'gru_win.txt', self.agent.state_dict()["rnn.weight_ih"].cpu().numpy()[128:, :])
        np.savetxt(path + 'gru_bir.txt', self.agent.state_dict()["rnn.bias_ih"].cpu().numpy()[:64])
        np.savetxt(path + 'gru_biz.txt', self.agent.state_dict()["rnn.bias_ih"].cpu().numpy()[64:128])
        np.savetxt(path + 'gru_bin.txt', self.agent.state_dict()["rnn.bias_ih"].cpu().numpy()[128:])

        np.savetxt(path + 'gru_whr.txt', self.agent.state_dict()["rnn.weight_hh"].cpu().numpy()[:64, :])
        np.savetxt(path + 'gru_whz.txt', self.agent.state_dict()["rnn.weight_hh"].cpu().numpy()[64:128, :])
        np.savetxt(path + 'gru_whn.txt', self.agent.state_dict()["rnn.weight_hh"].cpu().numpy()[128:, :])
        np.savetxt(path + 'gru_bhr.txt', self.agent.state_dict()["rnn.bias_hh"].cpu().numpy()[:64])
        np.savetxt(path + 'gru_bhz.txt', self.agent.state_dict()["rnn.bias_hh"].cpu().numpy()[64:128])
        np.savetxt(path + 'gru_bhn.txt', self.agent.state_dict()["rnn.bias_hh"].cpu().numpy()[128:])

        np.savetxt(path + 'fc2_weight.txt', self.agent.state_dict()["fc2.weight"].cpu().numpy())
        np.savetxt(path + 'fc2_bias.txt', self.agent.state_dict()["fc2.bias"].cpu().numpy())
