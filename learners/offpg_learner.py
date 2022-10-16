import copy
import torch as th
import numpy as np
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm

from .offpg_critic import OffPGCritic
from .qmixer import QMixer
from MARLLib.utils.buffer import EpisodeBatch
from MARLLib.utils.function import plot_compute_graph


class OffPGLearner:
    def __init__(self,
                 scheme,
                 n_actions,
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
                 tree_backup_step
                 ):

        self.n_actions = n_actions
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

        # model
        self.critic = OffPGCritic(scheme, n_actions, n_agents, critic_hidden_dim)
        self.target_critic = copy.deepcopy(self.critic)
        self.mixer = QMixer(n_agents, self.state_dim, mixing_embed_dim)
        self.target_mixer = copy.deepcopy(self.mixer)

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

        self.critic_training_count = 0
        self.last_critic_training_log_count = -1
        self.last_actor_training_log_step = -1

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

    def _build_td_lambda_targets(self, rewards, terminated, mask, target_q_total):
        """
        Args:
            rewards: (batch_size, episode_steps, 1)
            terminated: (batch_size, episode_steps, 1)
            mask: (batch_size, episode_steps, 1)
            target_q_total:

        Returns:
        """
        g_lambda = th.zeros_like(target_q_total)
        # 若不存在 terminated[t] == True, 超时或是成功，最后一步的 Q(t+1) 需要计算
        g_lambda[:, -1, :] = target_q_total[:, -1, :] * (1 - th.sum(terminated, dim=1))

        for t in range(rewards.shape[1]-1, -1, -1):
            # 有些 batch 很早就因为失败而停止了，terminated[t] == True 的那一步，g_lambda[t] 不应为 0，还有 r
            g_lambda[:, t, :] = self.td_lambda * self.gamma * g_lambda[:, t+1, :] + mask[:, t, :] * \
                                (rewards[:, t, :] + (1 - self.td_lambda) * self.gamma * target_q_total[:, t+1, :] *
                                 (1 - terminated[:, t, :]))

        return g_lambda[:, :-1, :]

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

    def deal_with_off_batch(self, off_batch):
        batch_size = off_batch.batch_size
        max_episode_length = off_batch.max_seq_length
        device = off_batch.device

        state = off_batch["state"]  # state: (batch_size, episode_steps, state_dim)
        avail_actions = off_batch["avail_actions"]
        obs = off_batch["obs"]
        actions = off_batch["actions"]  # actions: (batch_size, episode_steps, n_agents, 1)
        rewards = off_batch["reward"][:, :-1, :]
        terminated = off_batch["terminated"][:, :-1, :].float()
        mask = off_batch["filled"][:, :-1, :].float()
        mask[:, 1:, :] = mask[:, 1:, :] * (1 - terminated[:, :-1, :])

        # inputs: (batch_size, episode_steps, n_agents, state_dim+obs_dim+n_agents)
        inputs = self._build_critic_inputs(state, obs, batch_size, max_episode_length, device)

        self.controller.init_hidden(batch_size)
        # action_probs[i]: (batch_size, n_agents, n_actions)
        action_probs = [self.controller.forward(off_batch, t).detach() for t in range(max_episode_length)]
        # action_probs: (batch_size, max_episode_length, n_agents, n_actions)
        action_probs = th.stack(action_probs, dim=1)
        # 若 mask_before_softmax == True，无需以下操作
        action_probs[avail_actions == 0] = 0
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # -1: 最后一维
        action_probs[avail_actions == 0] = 0
        joint_action_prob = th.gather(action_probs, 3, actions).squeeze(3).prod(dim=2, keepdim=True)[:, :-1, :]

        target_q_locals = self.target_critic(inputs).detach()

        # 计算 expected_q_total
        expected_q_total = self.target_mixer(th.sum(target_q_locals * action_probs, dim=3), state, batch_size).detach()
        # expected_q_total[:, -1, :] = expected_q_total[:, -1, :] * (1 - th.sum(terminated, dim=1))
        expected_q_total[:, :-1, :] = expected_q_total[:, :-1, :] * mask

        # 计算 target_q_total
        target_q_locals = th.gather(target_q_locals, dim=3, index=actions).squeeze(3)
        target_q_total = self.target_mixer(target_q_locals, state, batch_size).detach()
        # target_q_total[:, -1, :] = target_q_total[:, -1, :] * (1 - th.sum(terminated, dim=1))
        target_q_total[:, :-1, :] = target_q_total[:, :-1, :] * mask

        # delta 对于每个 episode 的有效区间为: [0, terminated_step]
        delta = (rewards + self.gamma * expected_q_total[:, 1:, :] - target_q_total[:, :-1, :]) * mask

        tree_backup = th.zeros_like(delta)
        coefficient = 1.0
        tmp = delta
        padding = th.zeros_like(delta[:, :1, :])
        for _ in range(self.tree_backup_step):
            tree_backup += coefficient * tmp
            tmp = th.cat(((tmp * joint_action_prob)[:, 1:, :], padding), dim=1)
            coefficient *= self.gamma * self.td_lambda
        tree_backup += target_q_total[:, :-1, :]
        return inputs, state, actions, mask, tree_backup, max_episode_length

    def train_critic(self, on_batch: EpisodeBatch, off_batch: EpisodeBatch = None, critic_running_log=None):
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
        batch_size = on_batch.batch_size
        max_episode_length = on_batch.max_seq_length
        device = on_batch.device

        state = on_batch["state"]  # state: (batch_size, episode_steps, state_dim)
        # avail_actions = on_batch["avail_actions"]
        obs = on_batch["obs"]
        actions = on_batch["actions"]  # actions: (batch_size, episode_steps, n_agents, 1)
        rewards = on_batch["reward"][:, :-1, :]
        terminated = on_batch["terminated"][:, :-1, :].float()
        mask = on_batch["filled"][:, :-1, :].float()
        mask[:, 1:, :] = mask[:, 1:, :] * (1 - terminated[:, :-1, :])

        # inputs: (batch_size, episode_steps, n_agents, state_dim+obs_dim+n_agents)
        inputs = self._build_critic_inputs(state, obs, batch_size, max_episode_length, device)

        # 计算 target
        target_q_locals = self.target_critic(inputs).detach()
        target_q_locals = th.gather(target_q_locals, dim=3, index=actions).squeeze(3)
        target_q_total = self.target_mixer(target_q_locals, state, batch_size).detach()
        g_lambda = self._build_td_lambda_targets(rewards, terminated, mask, target_q_total).detach()

        # self.controller.init_hidden(batch_size)
        # # action_probs[i]: (batch_size, n_agents, n_actions)
        # action_probs = [self.controller.forward(on_batch, t) for t in range(max_episode_length)]
        # # action_probs: (batch_size, max_episode_length, n_agents, n_actions)
        # action_probs = th.stack(action_probs, dim=1)
        # # 若 mask_before_softmax == True，无需以下操作
        # # action_probs[avail_actions==0] = 0
        # # action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True) # -1: 最后一维
        # # action_probs[avail_actions==0] = 0
        # action_probs.detach()

        # QUESTION: 为何源代码中称 off_batch 为 best_batch?
        # 处理 off_policy 的部分
        if off_batch is not None:
            off_inputs, off_state, off_actions, off_mask, tree_backup, off_max_episode_length \
                = self.deal_with_off_batch(off_batch)
            inputs = th.cat((inputs, off_inputs), dim=0)
            state = th.cat((state, off_state), dim=0)
            actions = th.cat((actions, off_actions), dim=0)
            mask = th.cat((mask, off_mask), dim=0)
            g_lambda = th.cat((g_lambda, tree_backup), dim=0)
            max_episode_length = max(max_episode_length, off_max_episode_length)
            batch_size += off_batch.batch_size

        # train critic and mixer network
        for t in range(max_episode_length-1):
            mask_t = mask[:, t:t+1, :]  # 如果中间是 t 的话，就相当于是按下标取值了，这样就会少一维
            assert mask_t.sum() > 0.9, "max_episode_length is not correct!"

            # q_locals: (batch_size, 1, n_agents, n_actions)
            q_locals = self.critic(inputs[:, t:t+1, :, :])
            # q_locals_selected: (batch_size, 1, n_agents)
            q_locals_selected = th.gather(q_locals, dim=3, index=actions[:, t:t+1, :, :]).squeeze(3)
            q_total = self.mixer(q_locals_selected, state[:, t:t+1, :], batch_size)
            q_total_target = g_lambda[:, t:t+1, :]
            td_error = (q_total - q_total_target) * mask_t
            critic_loss = (td_error ** 2).sum() / mask_t.sum()

            # # 打印计算图
            # params_dict = dict()
            # params_dict.update(dict(self.controller.agent.named_parameters()))
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
            self.critic_training_count += 1

            # 记录训练过程的信息
            mask_num = mask_t.sum().item()
            critic_running_log["critic_loss"].append(critic_loss.item())
            critic_running_log["critic_grad_norm"].append(grad_norm)
            critic_running_log["td_error_abs"].append((td_error.abs().sum().item() / mask_num))
            critic_running_log["q_total_target_mean"].append((q_total_target * mask_t).sum().item() / mask_num)
            critic_running_log["q_total_mean"].append((q_total * mask_t).sum().item() / mask_num)
            critic_running_log["q_locals_max_mean"].append(
                (th.mean(q_locals.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_num)
            critic_running_log["q_locals_min_mean"].append(
                (th.mean(q_locals.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_num)
            critic_running_log["q_locals_max_var"].append(
                (th.var(q_locals.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_num)
            critic_running_log["q_locals_min_var"].append(
                (th.var(q_locals.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_num)

        if (self.critic_training_count - self.last_critic_training_log_count > self.target_update_interval) or \
                (self.last_critic_training_log_count == -1):
            # update target network
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.logger.info("critic_training_count: " + str(self.critic_training_count) + ", updated target network")
            self.last_critic_training_log_count = self.critic_training_count

    def train_actor(self, batch: EpisodeBatch, total_steps, critic_running_log=None):
        batch_size = batch.batch_size
        max_episode_length = batch.max_seq_length
        device = batch.device

        state = batch["state"]  # state: (batch_size, max_episode_length, state_dim)
        avail_actions = batch["avail_actions"][:, :-1, :]
        obs = batch["obs"]

        # 这里与 train_critic 不同的原因是为了方便 log_pi_selected 的计算
        actions = batch["actions"][:, :-1, :]  # actions: (batch_size, max_seq_length-1, n_agents, 1)
        terminated = batch["terminated"][:, :-1, :].float()
        mask = batch["filled"][:, :-1, :].float()
        # terminated == True: mask 填充到这一步位置（包括这一步）
        # 不存在 terminated == True: mask 填充到最后一步（max_seq_length-1）
        # Q 值通过 critic 网络得到，不需要考虑现实情况（terminated 了 Q 就得为零）
        mask[:, 1:, :] = mask[:, 1:, :] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).reshape(-1)

        # inputs: (batch_size, episode_steps, n_agents, state_dim+obs_dim+n_agents)
        inputs = self._build_critic_inputs(state, obs, batch_size, max_episode_length, device)
        q_locals = self.critic(inputs).detach()[:, :-1, :]

        # 通过 actor 网络得到 action probability
        self.controller.init_hidden(batch_size)
        action_probs = [self.controller.forward(batch, t) for t in range(max_episode_length-1)]
        action_probs = th.stack(action_probs, dim=1)
        # 若 mask_before_softmax == True，无需以下操作
        action_probs[avail_actions == 0] = 0
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # -1: 最后一维
        action_probs[avail_actions == 0] = 0

        # 计算 baseline: 一维向量, batch_size * max_episode_length * n_agents
        baseline = th.sum(action_probs*q_locals, dim=-1).reshape(-1).detach()

        # 计算 advantages: 一维向量, batch_size * max_episode_length * n_agents
        q_locals_selected = th.gather(q_locals, dim=3, index=actions).squeeze(3)
        advantages = (q_locals_selected.reshape(-1) - baseline).detach()

        # 计算 log_pi_selected
        pi = action_probs.reshape(-1, self.n_actions)  # (batch_size * max_episode_length * n_agents, n_actions)
        pi_selected = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)  # 一维向量
        pi_selected[mask == 0] = 1.0
        log_pi_selected = th.log(pi_selected)

        # 计算 coma loss
        coefficient = self.mixer.get_k(state[:, :-1, :], batch_size).reshape(-1)
        coma_loss = -(log_pi_selected * coefficient * advantages * mask).sum() / mask.sum()

        # # 打印计算图
        # params_dict = dict()
        # params_dict.update(dict(self.controller.agent.named_parameters()))
        # params_dict.update(dict(self.mixer.named_parameters()))
        # params_dict.update(dict(self.critic.named_parameters()))
        # plot_compute_graph(coma_loss, params_dict)

        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        # 限制梯度
        grad_norm = clip_grad_norm(self.agent_params, self.grad_norm_clip)
        self.agent_optimiser.step()

        if (total_steps - self.last_actor_training_log_step > self.learner_log_interval) or \
                (self.last_actor_training_log_step == -1):
            for key, value in critic_running_log.items():
                self.logger.log_stat(key, sum(value)/len(value), total_steps)
            # self.logger.log_stat("q_max_first", critic_running_log["q_max_first"], total_steps)
            # self.logger.log_stat("q_min_first", critic_running_log["q_min_first"], total_steps)
            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), total_steps)
            self.logger.log_stat("coma_loss", coma_loss.item(), total_steps)
            self.logger.log_stat("agent_grad_norm", grad_norm, total_steps)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), total_steps)

            self.last_actor_training_log_step = total_steps
