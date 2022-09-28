import torch as th

from rnn_agent import RNNAgent
from MARLLib.utils.action_selector import SELECTOR


class Controller:
    """为多个智能体提供 obs-->action 的映射"""

    def __init__(self,
                 scheme,
                 batch_size_run,
                 n_agents,
                 agent_output_type,
                 obs_last_action,
                 obs_agent_id,
                 mask_before_softmax,
                 rnn_hidden_dim,
                 n_actions,
                 action_selector,

                 epsilon_start,
                 epsilon_finish,
                 epsilon_anneal_time,
                 test_greedy):

        self.batch_size_run = batch_size_run
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
        self.action_selector = SELECTOR[action_selector](epsilon_start, epsilon_finish, epsilon_anneal_time,
                                                         test_greedy)
        self.hidden_states = None

    def init_hidden(self):
        self.hidden_states = self.controller.init_hidden().unsqueeze(0).unsqueeze(0) \
            .expand(self.batch_size_run, self.n_agents, -1)

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
            # epsilon-greedy
            if not test_mode:
                action_num = avail_actions_reshaped.sum(dim=1, keepdim=True).float() if self.mask_before_softmax \
                    else self.n_actions
                outputs = (1 - self.action_selector.epsilon) * outputs + \
                    self.action_selector.epsilon * th.ones_like(outputs) / action_num
                if self.mask_before_softmax:
                    outputs[avail_actions_reshaped == 0] = 0.0

        return outputs.view(ep_batch.batch_size, self.n_agents, -1)
