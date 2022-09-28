import torch as th
import torch.nn as nn
from torch.nn.functional import relu


class RNNAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, output_shape):
        super(RNNAgent, self).__init__()
        self.input_shape = input_shape
        self.rnn_hidden_dim = rnn_hidden_dim
        self.output_shape = output_shape

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, output_shape)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(self.rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        fc1_output = relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h_out = self.rnn(fc1_output, h_in)
        output = self.fc2(h_out)
        return output, h_out
