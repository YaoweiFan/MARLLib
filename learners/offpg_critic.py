import torch as th
from torch.nn.functional import relu


class OffPGCritic(th.nn.Module):

    def __init__(self,
                 scheme,
                 action_dim,
                 n_agents,
                 critic_hidden_dim,
                 ):
        super(OffPGCritic, self).__init__()

        self.n_agents = n_agents
        # state + obs + id
        input_shape = scheme["state"]["vshape"] + scheme["obs"]["vshape"] + n_agents
        self.fc = th.nn.Linear(input_shape, critic_hidden_dim)

        self.fc_v1 = th.nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.fc_v2 = th.nn.Linear(critic_hidden_dim, 1)

        self.fc_a1 = th.nn.Linear(critic_hidden_dim + action_dim, critic_hidden_dim)
        self.fc_a2 = th.nn.Linear(critic_hidden_dim, 1)

    def forward(self, vnet_inputs, actions):
        hidden = relu(self.fc(vnet_inputs))

        # fc_v1_output: (batch_size, episode_steps, n_agents, critic_hidden_dim)
        fc_v1_outputs = relu(self.fc_v1(hidden))
        v = self.fc_v2(fc_v1_outputs)

        # actions: (batch_size, episode_steps, n_agents, action_dim)
        # terminated = True 的那一步之后的一步的 action 在 rollout 过程中是会采集的，但这是不能用的
        combined = th.cat([hidden, actions], dim=3)
        fc_a1_outputs = relu(self.fc_a1(combined))
        a = self.fc_a2(fc_a1_outputs)

        return v+a, v  # Q(s, a), V(s)
