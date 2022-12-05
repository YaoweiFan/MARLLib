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
                 tree_backup_step
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

        # model
        self.critic = OffPGCritic(scheme, actions_dim, n_agents, critic_hidden_dim)
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
        obs = off_batch["obs"]
        actions = off_batch["actions"]  # actions: (batch_size, episode_steps, n_agents, 1)
        old_log_prob = off_batch["old_log_prob"][:, :-1, :]  # old_log_prob: (batch_size, episode_steps-1, n_agents, 1)
        rewards = off_batch["reward"][:, :-1, :]
        terminated = off_batch["terminated"][:, :-1, :].float()
        mask = off_batch["filled"][:, :-1, :].float()
        mask[:, 1:, :] = mask[:, 1:, :] * (1 - terminated[:, :-1, :])

        # 在 evaluate_actions 中 forward 得到 log_prob，buffer 中的没法用，因为是转换成 numpy 后存储的
        log_prob = []
        self.controller.init_hidden(batch_size)
        for t in range(max_episode_length - 1):
            log_prob.append(self.controller.evaluate_actions(off_batch, t).detach())
        # log_prob: (batch_size, episode_steps-1, n_agents, 1)
        log_prob = th.stack(log_prob, dim=1).squeeze(3).sum(dim=2, keepdim=True)

        # log_importance_weight: (batch_size, episode_steps-1, 1)
        log_importance_weight = log_prob - old_log_prob.squeeze(3).sum(dim=2, keepdim=True)
        c = th.min(th.ones_like(log_importance_weight), th.exp(log_importance_weight))

        # inputs: (batch_size, episode_steps, n_agents, state_dim+obs_dim+n_agents)
        inputs = self._build_critic_inputs(state, obs, batch_size, max_episode_length, device)
        target_q_locals, expected_q_locals = self.target_critic(inputs, actions)

        # 计算 expected_q_total
        expected_q_total = self.target_mixer(expected_q_locals.detach(), state, batch_size).detach()
        expected_q_total[:, -1, :] = expected_q_total[:, -1, :] * (1 - th.sum(terminated, dim=1))
        expected_q_total[:, :-1, :] = expected_q_total[:, :-1, :] * mask

        # 计算 target_q_total
        target_q_locals = target_q_locals.detach().squeeze(3)
        target_q_total = self.target_mixer(target_q_locals, state, batch_size).detach()
        # 不需要处理 target_q_total 最后一个 mask 与否，因为用不到
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
            tmp = th.cat(((tmp * c)[:, 1:, :], padding), dim=1)
            coefficient *= self.gamma * self.td_lambda
        tree_backup += target_q_total[:, :-1, :]
        return inputs, state, actions, mask, tree_backup, max_episode_length

    def train_critic(self, on_batch: EpisodeBatch, off_batch: EpisodeBatch, critic_running_log):
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
        obs = on_batch["obs"]
        actions = on_batch["actions"]  # actions: (batch_size, episode_steps, n_agents, action_dim)
        rewards = on_batch["reward"][:, :-1, :]  # rewards: (batch_size, episode_steps-1, 1)
        terminated = on_batch["terminated"][:, :-1, :].float()
        mask = on_batch["filled"][:, :-1, :].float()
        mask[:, 1:, :] = mask[:, 1:, :] * (1 - terminated[:, :-1, :])

        # inputs: (batch_size, episode_steps, n_agents, state_dim+obs_dim+n_agents)
        inputs = self._build_critic_inputs(state, obs, batch_size, max_episode_length, device)

        # 计算 target
        # target_q_locals: (batch_size, episode_steps, n_agents, 1)
        target_q_locals, _ = self.target_critic(inputs, actions)
        target_q_locals = target_q_locals.squeeze(3).detach()
        target_q_total = self.target_mixer(target_q_locals, state, batch_size).detach()
        g_lambda = self._build_td_lambda_targets(rewards, terminated, mask, target_q_total).detach()

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

            # q_locals: (batch_size, 1, n_agents)
            q_locals, _ = self.critic(inputs[:, t:t+1, :, :], actions[:, t:t+1, :, :])
            q_locals = q_locals.squeeze(3)
            q_total = self.mixer(q_locals, state[:, t:t+1, :], batch_size)
            q_total_target = g_lambda[:, t:t+1, :]
            td_error = (q_total - q_total_target) * mask_t
            critic_loss = (td_error ** 2).sum() / mask_t.sum()

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
            self.critic_training_count += 1

            # 记录训练过程的信息
            mask_num = mask_t.sum().item()
            critic_running_log["critic_loss"].append(critic_loss.item())
            critic_running_log["critic_grad_norm"].append(grad_norm)
            critic_running_log["td_error_abs"].append((td_error.abs().sum().item() / mask_num))
            critic_running_log["q_total_target_mean"].append((q_total_target * mask_t).sum().item() / mask_num)
            critic_running_log["q_total_mean"].append((q_total * mask_t).sum().item() / mask_num)
            critic_running_log["q_locals_mean"].append(
                (th.mean(q_locals, dim=2, keepdim=True) * mask_t).sum().item() / mask_num)
            critic_running_log["q_locals_var"].append(
                (th.var(q_locals, dim=2, keepdim=True) * mask_t).sum().item() / mask_num)

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
        obs = batch["obs"]

        # 这里与 train_critic 不同的原因是为了方便 log_pi_selected 的计算
        actions = batch["actions"]  # actions: (batch_size, max_seq_length-1, n_agents, 3)
        terminated = batch["terminated"][:, :-1, :].float()
        mask = batch["filled"][:, :-1, :].float()
        # terminated == True: mask 填充到这一步位置（包括这一步）
        # 不存在 terminated == True: mask 填充到最后一步（max_seq_length-1）
        # Q 值通过 critic 网络得到，不需要考虑现实情况（terminated 了 Q 就得为零）
        mask[:, 1:, :] = mask[:, 1:, :] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).reshape(-1)

        # inputs: (batch_size, episode_steps, n_agents, state_dim+obs_dim+n_agents)
        inputs = self._build_critic_inputs(state, obs, batch_size, max_episode_length, device)
        q_locals, expected_q_locals = self.critic(inputs, actions)
        q_locals = q_locals.detach().squeeze(3)[:, :-1, :]
        expected_q_locals = expected_q_locals.detach().squeeze(3)[:, :-1, :]

        # 计算 baseline: 一维向量, batch_size * max_episode_length-1 * n_agents
        baseline = expected_q_locals.reshape(-1)

        # 计算 advantages: 一维向量, batch_size * max_episode_length * n_agents
        advantages = q_locals.reshape(-1) - baseline

        # 在 evaluate_actions 中 forward 得到 log_prob，buffer 中的没法用，因为是转换成 numpy 后存储的
        log_prob = []
        self.controller.init_hidden(batch_size)
        for t in range(max_episode_length - 1):
            log_prob.append(self.controller.evaluate_actions(batch, t))
        # log_prob: (batch_size, episode_steps-1, n_agents, 1)
        log_prob = th.stack(log_prob, dim=1)
        log_prob = log_prob.reshape(-1)

        # 计算 coma loss
        coefficient = self.mixer.get_k(state[:, :-1, :], batch_size).reshape(-1)
        coma_loss = -(log_prob * coefficient * advantages * mask).sum() / mask.sum()

        # # 打印计算图
        # params_dict = dict()
        # params_dict.update(dict(self.controller.agent.named_parameters()))
        # params_dict.update({"log_std": self.controller.log_std})
        # params_dict.update(dict(self.mixer.named_parameters()))
        # params_dict.update(dict(self.critic.named_parameters()))
        # plot_compute_graph(coma_loss, params_dict)

        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        # 限制梯度
        grad_norm = clip_grad_norm(self.agent_params, self.grad_norm_clip)
        self.agent_optimiser.step()

        critic_running_log["log_std"].append(self.controller.log_std[0].item())

        if (total_steps - self.last_actor_training_log_step > self.learner_log_interval) or \
                (self.last_actor_training_log_step == -1):
            for key, value in critic_running_log.items():
                self.logger.log_stat(key, sum(value)/len(value), total_steps)
            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), total_steps)
            self.logger.log_stat("coma_loss", coma_loss.item(), total_steps)
            self.logger.log_stat("agent_grad_norm", grad_norm, total_steps)
            self.logger.log_stat("log_prob", (log_prob * mask).sum().item() / mask.sum().item(), total_steps)

            self.last_actor_training_log_step = total_steps
