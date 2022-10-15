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
        self.fc1 = th.nn.Linear(input_shape, critic_hidden_dim)
        self.fc2 = th.nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.fc_v = th.nn.Linear(critic_hidden_dim, 1)
        self.fc_a = th.nn.Linear(critic_hidden_dim+action_dim, 1)

    def forward(self, vnet_inputs, actions):
        fc1_output = relu(self.fc1(vnet_inputs))
        # fc2_output: (batch_size, episode_steps, n_agents, critic_hidden_dim)
        fc2_output = relu(self.fc2(fc1_output))
        v = self.fc_v(fc2_output)
        # actions: (batch_size, episode_steps, n_agents, action_dim)
        # terminated = True 的那一步之后的一步的 action 在 rollout 过程中是会采集的，但这是不能用的
        fc_a_inputs = th.cat([fc2_output, actions], dim=3)
        a = self.fc_a(fc_a_inputs)
        return v+a, v  # Q(s, a), Q(s)
