import os
import torch as th
import numpy as np

from rnn_agent import RNNAgent
from action_selector import MultinomialActionSelector


class Controller:
    """为多个智能体提供 obs-->action 的映射"""

    def __init__(self,
                 scheme,
                 n_agents,
                 agent_output_type,
                 obs_last_action,
                 obs_agent_id,
                 mask_before_softmax,
                 rnn_hidden_dim,
                 n_actions,

                 epsilon_start,
                 epsilon_finish,
                 epsilon_anneal_time,
                 test_greedy):

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_output_type = agent_output_type
        self.obs_last_action = obs_last_action
        self.obs_agent_id = obs_agent_id
        self.mask_before_softmax = mask_before_softmax
        # 计算输入维度
        input_shape = scheme["obs"]["vshape"]
        if obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if obs_agent_id:
            input_shape += self.n_agents
        # 创建 controller
        self.controller = RNNAgent(input_shape, rnn_hidden_dim, n_actions)
        # 创建 action selector
        self.action_selector = MultinomialActionSelector(epsilon_start, epsilon_finish, epsilon_anneal_time,
                                                         test_greedy)
        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = self.controller.init_hidden().unsqueeze(0).unsqueeze(0) \
            .expand(batch_size, self.n_agents, -1)

    def select_actions(self, ep_batch, episode_step, steps, avail_env=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, episode_step]
        outputs = self.forward(ep_batch, episode_step, test_mode)
        return self.action_selector.select_actions(outputs[avail_env], avail_actions[avail_env], steps, test_mode)

    def _build_inputs(self, ep_batch, episode_step):
        """
        inputs:
               obs: (batch_size_run, n_agents, obs_size)
               last_action: (batch_size_run, n_agents, last_action_size)
               obs_agent_id: (batch_size_run, n_agents, obs_agent_id_size)
        """
        inputs = [ep_batch["obs"][:, episode_step]]
        if self.obs_last_action:
            if episode_step == 0:
                inputs.append(th.zeros_like(ep_batch["action_onehot"][:, episode_step]))
            else:
                inputs.append(ep_batch["action_onehot"][:, episode_step - 1])
        if self.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).expand(ep_batch.batch_size, -1, -1))
        inputs = th.cat([item.reshape(ep_batch.batch_size * self.n_agents, -1) for item in inputs], dim=1)
        return inputs

    def forward(self, ep_batch, episode_step, test_mode=False):
        inputs = self._build_inputs(ep_batch, episode_step)
        avail_actions = ep_batch["avail_actions"][:, episode_step]
        # outputs: (batch_size_run * n_agents, n_actions)
        outputs, self.hidden_states = self.controller(inputs, self.hidden_states)

        # softmax
        if self.agent_output_type == "pi_logits":
            avail_actions_reshaped = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
            if self.mask_before_softmax:
                # 将不可行的动作对应的输出设置成负值，这样在 softmax 之后这些输出对应的概率值就会非常小
                outputs[avail_actions_reshaped == 0] = -1e11

            outputs = th.nn.functional.softmax(outputs, dim=-1)

            # TODO: selector 中已包含 epsilon-greedy，这部分是否有必要？
            # ANSWER: 需要反向梯度传导，有必要
            #         这部分与 selector 中的 epsilon-greedy 实现含义完全相同（评估策略和行为策略完全一致）
            #         但拿这里的输出再作为 selector_actions 的输入，相当于进行了两次 epsilon-greedy, 是否有误？
            # epsilon-greedy
            if not test_mode:
                action_num = avail_actions_reshaped.sum(dim=1, keepdim=True).float() if self.mask_before_softmax \
                    else self.n_actions
                outputs = (1 - self.action_selector.epsilon) * outputs + \
                    self.action_selector.epsilon * th.ones_like(outputs) / action_num
                if self.mask_before_softmax:
                    outputs[avail_actions_reshaped == 0] = 0.0

        return outputs.view(ep_batch.batch_size, self.n_agents, -1)

    def cuda(self):
        self.controller.cuda()

    def save_models(self, path):
        th.save(self.controller.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path, record_param=False):
        self.controller.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if record_param:
            self.record(os.path.join(path, "parameters"))
            raise Exception("Controller network loaded successfully!")

    def record(self, path):
        # 记录 controller 网络参数
        np.savetxt(path + 'fc1_weight.txt', self.controller.state_dict()["fc1.weight"].cpu().numpy())
        np.savetxt(path + 'fc1_bias.txt', self.controller.state_dict()["fc1.bias"].cpu().numpy())

        np.savetxt(path + 'gru_wir.txt', self.controller.state_dict()["rnn.weight_ih"].cpu().numpy()[:64, :])
        np.savetxt(path + 'gru_wiz.txt', self.controller.state_dict()["rnn.weight_ih"].cpu().numpy()[64:128, :])
        np.savetxt(path + 'gru_win.txt', self.controller.state_dict()["rnn.weight_ih"].cpu().numpy()[128:, :])
        np.savetxt(path + 'gru_bir.txt', self.controller.state_dict()["rnn.bias_ih"].cpu().numpy()[:64])
        np.savetxt(path + 'gru_biz.txt', self.controller.state_dict()["rnn.bias_ih"].cpu().numpy()[64:128])
        np.savetxt(path + 'gru_bin.txt', self.controller.state_dict()["rnn.bias_ih"].cpu().numpy()[128:])

        np.savetxt(path + 'gru_whr.txt', self.controller.state_dict()["rnn.weight_hh"].cpu().numpy()[:64, :])
        np.savetxt(path + 'gru_whz.txt', self.controller.state_dict()["rnn.weight_hh"].cpu().numpy()[64:128, :])
        np.savetxt(path + 'gru_whn.txt', self.controller.state_dict()["rnn.weight_hh"].cpu().numpy()[128:, :])
        np.savetxt(path + 'gru_bhr.txt', self.controller.state_dict()["rnn.bias_hh"].cpu().numpy()[:64])
        np.savetxt(path + 'gru_bhz.txt', self.controller.state_dict()["rnn.bias_hh"].cpu().numpy()[64:128])
        np.savetxt(path + 'gru_bhn.txt', self.controller.state_dict()["rnn.bias_hh"].cpu().numpy()[128:])

        np.savetxt(path + 'fc2_weight.txt', self.controller.state_dict()["fc2.weight"].cpu().numpy())
        np.savetxt(path + 'fc2_bias.txt', self.controller.state_dict()["fc2.bias"].cpu().numpy())
