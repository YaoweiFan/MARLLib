import copy
import torch as th
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm

from .offpg_critic import OffPGCritic
from MARLLib.utils.buffer import EpisodeBatch
from MARLLib.utils.function import plot_compute_graph


class OffPGLearner:
    def __init__(self,
                 scheme,
                 actions_dim,
                 n_agents,
                 critic_hidden_dim,
                 controller,
                 logger,
                 actor_learning_rate,
                 critic_learning_rate,
                 optim_alpha,
                 optim_eps,
                 gamma,
                 td_lambda,
                 grad_norm_clip,
                 target_update_interval,
                 learner_log_interval,
                 tree_backup_step,
                 soft_update_alpha,
                 ):

        self.actions_dim = actions_dim
        self.n_agents = n_agents
        self.controller = controller
        self.logger = logger
        self.state_dim = scheme["state"]["vshape"]
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.grad_norm_clip = grad_norm_clip
        self.target_update_interval = target_update_interval
        self.learner_log_interval = learner_log_interval
        self.tree_backup_step = tree_backup_step
        self.soft_update_alpha = soft_update_alpha

        # model
        self.critic = OffPGCritic(scheme, actions_dim, n_agents, critic_hidden_dim)
        self.target_critic = copy.deepcopy(self.critic)

        self.target_controller = copy.deepcopy(self.controller)
        # self.target_controller.cuda()

        # optimiser
        self.agent_params = list(self.controller.parameters())
        self.agent_optimiser = RMSprop(params=self.agent_params, lr=actor_learning_rate, alpha=optim_alpha,
                                       eps=optim_eps)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=critic_learning_rate, alpha=optim_alpha,
                                        eps=optim_eps)

        self.training_count = 0
        self.last_training_log_step = -1

    def _build_critic_inputs(self, state, obs, batch_size, max_t, device):
        """
        Args:
            state: (batch_size, episode_steps, state_dim)
            obs: (batch_size, episode_steps, n_agents, obs_dim)

        Returns:
            inputs: (batch_size, episode_steps, n_agents, state_dim+obs_dim+n_agents)
        """
        inputs = [state.unsqueeze(2).repeat(1, 1, self.n_agents, 1),
                  obs,
                  th.eye(self.n_agents, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, max_t, -1, -1)]
        inputs = th.cat([item.reshape(batch_size, max_t, self.n_agents, -1) for item in inputs], dim=-1)
        return inputs

    def cuda(self):
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # self.target_critic.load_state_dict(self.critic.agent.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

    def train(self, off_batch: EpisodeBatch, total_steps, training_log=None):
        """
        pre_transition_data = {                             pre_transition_data = {
            "state": [],                                        "state": [],
            "avail_actions": [],                                "avail_actions": [],
            "obs": []                                           "obs": []
        }                                                   }
        actions                                     ......
        post_transition_data = {
            "reward": [],
            "terminated": []
        }
        """
        batch_size = off_batch.batch_size
        max_episode_length = off_batch.max_seq_length
        device = off_batch.device

        state = off_batch["state"]  # state: (batch_size, episode_steps, state_dim)
        obs = off_batch["obs"]
        actions = off_batch["actions"]  # actions: (batch_size, episode_steps, n_agents, action_dim)
        rewards = off_batch["reward"][:, :-1, :, :]  # rewards: (batch_size, episode_steps, n_agents, 1)
        terminated = off_batch["terminated"].float()
        mask = off_batch["filled"].float()
        mask[:, 1:, :] = mask[:, 1:, :] * (1 - terminated[:, :-1, :])
        mask_num = mask[:, :-1, :].sum()

        # inputs: (batch_size, episode_steps, n_agents, state_dim+obs_dim+n_agents)
        inputs = self._build_critic_inputs(state, obs, batch_size, max_episode_length, device)

        # target_actor --> target_critic
        target_action = []
        for t in range(max_episode_length):
            target_action.append(self.target_controller.forward(off_batch, t).detach())
        # target_actions: (batch_size, episode_steps, n_agents, action_dim)
        target_actions = th.stack(target_action, dim=1)
        # target_q_locals: (batch_size, episode_steps, n_agents, 1)
        target_q_locals = self.target_critic(inputs, target_actions).detach()
        target_q_locals = target_q_locals.squeeze(3)
        # 无 terminated 的都有效，有 terminated 的，terminated 后一步无效
        target_q_locals[:, :, :1] = target_q_locals[:, :, :1] * mask
        target_q_locals[:, :, 1:2] = target_q_locals[:, :, 1:2] * mask

        # actor --> critic, rollout 得到的 actions 用在此处，所以不需要使用 actor
        q_locals = self.critic(inputs, actions)
        q_locals = q_locals.squeeze(3)
        # terminated 的那一步是有效的，这一步的 r 要利用的
        q_locals[:, :, :1] = q_locals[:, :, :1] * mask
        q_locals[:, :, 1:2] = q_locals[:, :, 1:2] * mask

        # td_error 对于每个 episode 的有效区间为: [0, terminated_step]
        td_error_0 = rewards[:, :, 0, :] * mask[:, :-1, :] + self.gamma * target_q_locals[:, 1:, :1] - q_locals[:, :-1, :1]
        td_error_1 = rewards[:, :, 1, :] * mask[:, :-1, :] + self.gamma * target_q_locals[:, 1:, 1:2] - q_locals[:, :-1, 1:2]

        critic_loss = ((td_error_0 ** 2).sum() + (td_error_1 ** 2).sum()) / (2 * mask_num)

        # # 打印计算图
        # params_dict = dict()
        # params_dict.update(dict(self.controller.agent.named_parameters()))
        # params_dict.update({"log_std": self.controller.log_std})
        # params_dict.update(dict(self.critic.named_parameters()))
        # plot_compute_graph(critic_loss, params_dict)

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        # 限制梯度
        grad_norm = clip_grad_norm(self.critic_params, self.grad_norm_clip)
        self.critic_optimiser.step()

        # 记录 critic 训练过程的信息
        training_log["critic_loss"].append(critic_loss.item())
        training_log["critic_grad_norm"].append(grad_norm)
        training_log["td_error_0_abs"].append(td_error_0.abs().sum().item() / mask_num.item())
        training_log["td_error_1_abs"].append(td_error_1.abs().sum().item() / mask_num.item())
        training_log["q_locals_mean"].append(
            (th.mean(q_locals, dim=2, keepdim=True) * mask).sum().item() / mask_num.item())
        training_log["q_locals_var"].append(
            (th.var(q_locals, dim=2, keepdim=True) * mask).sum().item() / mask_num.item())

        training_log["action_x_mean"].append(
            (th.mean(actions[:, :, :, 0], dim=2, keepdim=True) * mask).sum().item() / mask_num.item())
        training_log["action_y_mean"].append(
            (th.mean(actions[:, :, :, 1], dim=2, keepdim=True) * mask).sum().item() / mask_num.item())

        self.training_count += 1

        # actor --> critic
        action = []
        for t in range(max_episode_length):
            action.append(self.controller.forward(off_batch, t))
        # actions: (batch_size, episode_steps, n_agents, action_dim)
        actions = th.stack(action, dim=1)
        # q_locals: (batch_size, episode_steps, n_agents, 1)
        q_locals = self.critic(inputs, actions)
        q_locals = q_locals.squeeze(3)

        q_locals[:, :, :1] = q_locals[:, :, :1] * mask  # 无 terminated 的都有效，有 terminated 的，terminated 后一步无效
        q_locals[:, :, 1:2] = q_locals[:, :, 1:2] * mask
        actor_loss = -q_locals.sum() / (2 * mask_num)

        # # 打印计算图
        # params_dict = dict()
        # params_dict.update(dict(self.controller.agent.named_parameters()))
        # params_dict.update({"log_std": self.controller.log_std})
        # params_dict.update(dict(self.critic.named_parameters()))
        # plot_compute_graph(actor_loss, params_dict)

        self.agent_optimiser.zero_grad()
        actor_loss.backward()
        # 限制梯度
        grad_norm = clip_grad_norm(self.agent_params, self.grad_norm_clip)
        self.agent_optimiser.step()

        # 一次训练后减小探索方差，target_controller 不必更新这一项，soft update 的时候同步即可
        self.controller.variance_reduce()

        # 记录 actor 训练过程的信息
        training_log["actor_loss"].append(actor_loss.item())
        training_log["agent_grad_norm"].append(grad_norm)
        training_log["log_std"].append(self.controller.log_std[0].item())

        # 每经过一定的训练次数，打印训练信息
        if (total_steps - self.last_training_log_step > self.learner_log_interval) or \
                (self.last_training_log_step == -1):
            for key, value in training_log.items():
                self.logger.log_stat(key, sum(value)/len(value), total_steps)
            self.last_training_log_step = total_steps

        if self.training_count % 100 == 0:
            # 每经过一定的训练次数，soft update target network
            self.target_critic.soft_update(self.critic, self.soft_update_alpha)
            self.target_controller.soft_update(self.controller, self.soft_update_alpha)
            self.logger.info("training_count: " + str(self.training_count) + ", target network soft updated")
