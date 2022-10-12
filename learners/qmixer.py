import torch as th


class QMixer(th.nn.Module):
    def __init__(self,
                 n_agents,
                 state_dim,
                 mixing_embed_dim,
                 ):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        self.hyper_w1 = th.nn.Linear(self.state_dim, self.mixing_embed_dim*self.n_agents)
        self.hyper_w2 = th.nn.Linear(self.state_dim, self.mixing_embed_dim)

    def get_k(self, states, batch_size):
        w1 = th.abs(self.hyper_w1(states))
        w1 = w1.reshape(-1, self.n_agents, self.mixing_embed_dim)
        w2 = th.abs(self.hyper_w2(states))
        w2 = w2.reshape(-1, self.mixing_embed_dim, 1)
        k = th.bmm(w1, w2).view(batch_size, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        # IMPROVED: 比源代码多了个 detach，应该是合理的
        return k.detach()

    def forward(self, q_locals, states, batch_size):
        """
        q_locals: (batch_size, episode_steps, n_agents)
        states: (batch_size, episode_steps, state_dim)
        """
        states = states.reshape(-1, self.state_dim)
        q_locals = q_locals.reshape(-1, 1, self.n_agents)

        # k: (batch_size*episode_steps, self.n_agents, 1)
        w1 = th.abs(self.hyper_w1(states))
        w1 = w1.reshape(-1, self.n_agents, self.mixing_embed_dim)
        w2 = th.abs(self.hyper_w2(states))
        w2 = w2.reshape(-1, self.mixing_embed_dim, 1)
        k = th.bmm(w1, w2)
        k = k / th.sum(k, dim=1, keepdim=True)

        # IMPROVING: 只有k，缺少 b
        return th.bmm(q_locals, k).reshape(batch_size, -1, 1)
