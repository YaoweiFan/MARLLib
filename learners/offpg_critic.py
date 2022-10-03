import torch as th
from torch.nn.functional import relu


class OffPGCritic(th.nn.Module):

    def __init__(self,
                 scheme,
                 n_actions,
                 n_agents,
                 critic_hidden_dim,
                 ):
        super(OffPGCritic, self).__init__()

        self.n_actions = n_actions
        self.n_agents = n_agents
        # state + obs + id
        input_shape = scheme["state"]["vshape"] + scheme["obs"]["vshape"] + n_agents
        self.fc1 = th.nn.Linear(input_shape, critic_hidden_dim)
        self.fc2 = th.nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.fc_v = th.nn.Linear(critic_hidden_dim, 1)
        self.fc_a = th.nn.Linear(critic_hidden_dim, n_actions)

    def forward(self, inputs):
        fc1_output = relu(self.fc1(inputs))
        fc2_output = relu(self.fc2(fc1_output))
        v = self.fc_v(fc2_output)
        a = self.fc_a(fc2_output)
        return v+a  # Q(s, a)
