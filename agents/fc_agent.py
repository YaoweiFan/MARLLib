import torch.nn as nn
from torch.nn.functional import relu


class FcAgent(nn.Module):
    def __init__(self, input_shape, actor_hidden_dim, output_shape):
        super(FcAgent, self).__init__()
        self.fc1 = nn.Linear(input_shape, actor_hidden_dim)
        self.fc2 = nn.Linear(actor_hidden_dim, actor_hidden_dim)
        self.fc3 = nn.Linear(actor_hidden_dim, output_shape)

    def forward(self, inputs):
        fc1_output = relu(self.fc1(inputs))
        fc2_output = relu(self.fc2(fc1_output))
        output = self.fc3(fc2_output)
        return output

    def soft_update(self, source, alpha):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - alpha) * target_param.data + alpha * source_param.data)
