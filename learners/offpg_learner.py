import copy
import torch as th
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm

from .offpg_critic import OffPGCritic
from .qmixer import QMixer
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
                 mixing_embed_dim,
                 actor_learning_rate,
                 critic_learning_rate,
                 mixer_learning_rate,
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
        self.mixer = QMixer(n_agents, self.state_dim, mixing_embed_dim)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_controller = copy.deepcopy(self.controller)
        # self.target_controller.cuda()

        # optimiser
        self.agent_params = list(self.controller.parameters())
        self.agent_optimiser = RMSprop(params=self.agent_params, lr=actor_learning_rate, alpha=optim_alpha,
                                       eps=optim_eps)
        critic_params = list(self.critic.parameters())
        self.critic_optimiser = RMSprop(params=critic_params, lr=critic_learning_rate, alpha=optim_alpha,
                                        eps=optim_eps)
        mixer_params = list(self.mixer.parameters())
        self.mixer_optimiser = RMSprop(params=mixer_params, lr=mixer_learning_rate, alpha=optim_alpha,
                                       eps=optim_eps)
        self.critic_and_mixer_params = critic_params + mixer_params

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
        self.mixer.cuda()
        self.target_critic.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.mixer_optimiser.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # self.target_critic.load_state_dict(self.critic.agent.state_dict())
        # self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimiser.load_state_dict(
            th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))

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
        rewards = off_batch["reward"][:, :-1, :]
        terminated = off_batch["terminated"].float()
        mask = off_batch["filled"].float()
        mask[:, 1:, :] = mask[:, 1:, :] * (1 - terminated[:, :-1, :])
        mask_num = mask[:, :-1, :].sum()

        # inputs: (batch_size, episode_steps, n_agents, state_dim+obs_dim+n_agents)
        inputs = self._build_critic_inputs(state, obs, batch_size, max_episode_length, device)

        # target_actor --> target_critic
        target_action = []
        for t in range(max_episode_length):
            target_action.append(self.target_controller.forward(off_batch, t, deterministic=True).detach())
        # target_actions: (batch_size, episode_steps, n_agents, action_dim)
        target_actions = th.stack(target_action, dim=1)
        # target_q_locals: (batch_size, episode_steps, n_agents, 1)
        target_q_locals = self.target_critic(inputs, target_actions).detach()
        # target_q_total: (batch_size, episode_steps, 1)
        target_q_locals = target_q_locals.squeeze(3)
        target_q_total = self.target_mixer(target_q_locals, state, batch_size).detach()
        target_q_total = target_q_total * mask  # 无 terminated 的都有效，有 terminated 的，terminated 后一步无效

        # actor --> critic, rollout 得到的 actions 用在此处，所以不需要使用 actor
        q_locals = self.critic(inputs, actions)
        q_locals = q_locals.squeeze(3)
        q_total = self.mixer(q_locals, state, batch_size)
        q_total = q_total * mask  # terminated 的那一步是有效的，这一步的 r 要利用的

        # td_error 对于每个 episode 的有效区间为: [0, terminated_step]
        td_error = rewards * mask[:, :-1, :] + self.gamma * target_q_total[:, 1:, :] - q_total[:, :-1, :]
        critic_loss = (td_error ** 2).sum() / mask_num

        # # 打印计算图
        # params_dict = dict()
        # params_dict.update(dict(self.controller.agent.named_parameters()))
        # params_dict.update({"log_std": self.controller.log_std})
        # params_dict.update(dict(self.mixer.named_parameters()))
        # params_dict.update(dict(self.critic.named_parameters()))
        # plot_compute_graph(critic_loss, params_dict)

        # IMPROVING: 源代码中这里还有一个 goal_loss，不知道是干什么用的，源代码也没有用上这一项
        self.critic_optimiser.zero_grad()
        self.mixer_optimiser.zero_grad()
        critic_loss.backward()
        # 限制梯度
        grad_norm = clip_grad_norm(self.critic_and_mixer_params, self.grad_norm_clip)
        self.critic_optimiser.step()
        self.mixer_optimiser.step()

        # 记录 critic 训练过程的信息
        training_log["critic_loss"].append(critic_loss.item())
        training_log["critic_grad_norm"].append(grad_norm)
        training_log["td_error_abs"].append(td_error.abs().sum().item() / mask_num.item())
        training_log["target_q_total_mean"].append(target_q_total.sum().item() / mask_num.item())
        training_log["q_total_mean"].append(q_total.sum().item() / mask_num.item())
        training_log["q_locals_mean"].append(
            (th.mean(q_locals, dim=2, keepdim=True) * mask).sum().item() / mask_num.item())
        training_log["q_locals_var"].append(
            (th.var(q_locals, dim=2, keepdim=True) * mask).sum().item() / mask_num.item())

        training_log["action_x_mean"].append(
            (th.mean(actions[:, :, :, 0], dim=2, keepdim=True) * mask).sum().item() / mask_num.item())
        training_log["action_y_mean"].append(
            (th.mean(actions[:, :, :, 1], dim=2, keepdim=True) * mask).sum().item() / mask_num.item())
        training_log["action_z_mean"].append(
            (th.mean(actions[:, :, :, 2], dim=2, keepdim=True) * mask).sum().item() / mask_num.item())

        self.training_count += 1

        # actor --> critic
        action = []
        for t in range(max_episode_length):
            action.append(self.controller.forward(off_batch, t, deterministic=True))
        # actions: (batch_size, episode_steps, n_agents, action_dim)
        actions = th.stack(action, dim=1)
        # q_locals: (batch_size, episode_steps, n_agents, 1)
        q_locals = self.critic(inputs, actions)
        # q_total: (batch_size, episode_steps, 1)
        q_locals = q_locals.squeeze(3)
        q_total = self.mixer(q_locals, state, batch_size)
        q_total = q_total * mask  # 无 terminated 的都有效，有 terminated 的，terminated 后一步无效
        actor_loss = -q_total.sum() / mask_num

        # # 打印计算图
        # params_dict = dict()
        # params_dict.update(dict(self.controller.agent.named_parameters()))
        # params_dict.update({"log_std": self.controller.log_std})
        # params_dict.update(dict(self.mixer.named_parameters()))
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

        if self.training_count % 3 == 0:
            # 每经过一定的训练次数，soft update target network
            self.target_critic.soft_update(self.critic, self.soft_update_alpha)
            self.target_mixer.soft_update(self.mixer, self.soft_update_alpha)
            self.target_controller.soft_update(self.controller, self.soft_update_alpha)
            self.logger.info("training_count: " + str(self.training_count) + ", updated target network")
