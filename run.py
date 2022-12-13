import os
from os.path import dirname, abspath
import time
import torch as th
from types import SimpleNamespace
import pprint

from MARLLib.runners import RUNNER
from MARLLib.agents.controller import Controller
from MARLLib.learners.offpg_learner import OffPGLearner
from MARLLib.utils.logging import Logger
from MARLLib.utils.buffer import ReplayBuffer
from MARLLib.utils.timehelper import time_left, time_str


def config_sanity_check_and_adjust(config, log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")
    config["device"] = "cuda" if config["use_cuda"] else "cpu"

    # test_n_episodes 应该是 batch_size_run 的整数倍
    if config["test_n_episodes"] < config["batch_size_run"]:
        config["test_n_episodes"] = config["batch_size_run"]
    else:
        config["test_n_episodes"] = (config["test_n_episodes"] // config["batch_size_run"]) * config["batch_size_run"]


def evaluate_only(args, runner):
    for i in range(args.evaluate_args.evaluate_n_episodes):
        runner.set_path_name()
        runner.set_steps(runner.steps)
        runner.run()
    runner.log()
    runner.close_env()


def run_sequential(args, logger):
    # 选择 ParallelRunner 或者是 EpisodeRunner
    runner = RUNNER[args.runner](logger, args.device, args.batch_size_run, args.clip_obs, args.clip_state, args.epsilon,
                                 args.use_running_normalize, args.test_n_episodes, args.runner_log_interval, args.env,
                                 args.env_args)
    env_info = runner.get_env_info()

    # 创建 on_policy buffer 和 off_policy buffer
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": env_info["action_dim"], "group": "agents"},
        "old_log_prob": {"vshape": (1,), "group": "agents"},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": env_info["n_agents"]  # 如果是每个 agent 都拥有一份，shape 就应该增加一维
    }
    preprocess = {}

    on_buffer = ReplayBuffer(scheme, groups, args.on_buffer_size, env_info["episode_limit"]+1,
                             preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)
    on_batch_size = args.on_batch_size

    # 创建 controller
    controller = Controller(on_buffer.scheme, env_info["n_agents"], args.obs_last_action, args.obs_agent_id,
                            args.rnn_hidden_dim, env_info["action_dim"], args.log_std_init)

    # setup runner
    runner.setup(scheme, groups, preprocess, controller)

    # 创建 learner
    learner = OffPGLearner(on_buffer.scheme, env_info["action_dim"], env_info["n_agents"], args.critic_hidden_dim,
                           controller, logger, args.mixing_embed_dim, args.actor_learning_rate,
                           args.critic_learning_rate, args.mixer_learning_rate, args.optim_alpha, args.optim_eps,
                           args.gamma, args.td_lambda, args.grad_norm_clip, args.target_update_interval,
                           args.learner_log_interval)

    # use cuda
    if args.use_cuda:
        controller.cuda()
        learner.cuda()

    # 从最近的 checkpoint 恢复
    if args.checkpoint_path != "":
        checkpoint_path = os.path.join(dirname(dirname(__file__)), args.checkpoint_path)
        if not os.path.isdir(checkpoint_path):
            logger.info("Checkpoint directory {} doesn't exist".format(checkpoint_path))
            return

        points = []
        for name in os.listdir(checkpoint_path):
            full_name = os.path.join(checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                points.append(int(name))

        # 选择最后生成的 model 或是 离 load_point 最近的 model
        load_point = max(points) if args.load_step == 0 else min(points, key=lambda x: abs(x - args.load_step))
        model_path = os.path.join(checkpoint_path, str(load_point))
        logger.info("Loading model from {}".format(model_path))
        controller.load_models(model_path, args.record_param)
        learner.load_models(model_path)
        runner.load_normalizer(model_path)

        runner.steps = load_point

    # 仅仅是 evaluate policy
    if args.evaluate:
        assert args.checkpoint_path != "", "evaluate but the model path is empty!"
        evaluate_only(args, runner)
        return

    # 开始训练
    episode_num = 0
    last_test_timestep = 0
    last_log_timestep = 0
    last_model_save_timestep = 0

    start_time = time.time()  # 训练开始时间
    last_time = start_time  # 上一次 test 时间

    logger.info("Beginning training for {} timesteps ...".format(args.t_max))
    while runner.steps <= args.t_max:
        critic_running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "q_total_target_mean": [],
            "q_total_mean": [],
            "q_locals_mean": [],
            "q_locals_var": [],
            "log_std": []
        }

        # rollout， 每个子进程走完一个 episode
        episode_batch = runner.rollout(test_mode=False)
        on_buffer.insert_episode_batch(episode_batch)

        # train
        if on_buffer.can_sample(on_batch_size):
            # train critic
            on_buffer_samples = on_buffer.uni_sample(on_batch_size)
            on_buffer_samples.to(args.device)
            # 获得 samples 中最长 episode 的长度
            max_episode_length = on_buffer_samples.max_t_filled()
            learner.train_critic(on_buffer_samples[:, :max_episode_length], critic_running_log)

            # train actor
            # 只选取用当前策略跑出来的 on_batch_size 个 episode，确保 on_policy 训练 actor 
            latest_samples = on_buffer.sample_latest(on_batch_size)
            latest_samples.to(args.device)
            max_episode_length = latest_samples.max_t_filled()
            learner.train_actor(latest_samples[:, :max_episode_length], runner.steps, critic_running_log)

            # 减小 action variance
            controller.variance_reduce()

        # test
        if (runner.steps - last_test_timestep > args.test_interval) or (last_test_timestep == 0):
            logger.info("Total steps: {} / {}".format(runner.steps, args.t_max))
            logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_timestep, runner.steps, args.t_max),
                time_str(time.time() - start_time))
            )
            last_time = time.time()
            last_test_timestep = runner.steps
            for _ in range(args.test_n_episodes // args.batch_size_run):
                runner.rollout(test_mode=True)

        # save
        if args.save_model and \
                (runner.steps - last_model_save_timestep > args.save_model_interval) or (last_model_save_timestep == 0):
            last_model_save_timestep = runner.steps
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.steps))
            os.makedirs(save_path, exist_ok=True)
            logger.info("Saving models to {}".format(save_path))

            controller.save_models(save_path)  # agent
            learner.save_models(save_path)  # critic and mixer
            runner.save_normalizer(save_path)  # normalizer

        episode_num += args.batch_size_run
        if (runner.steps - last_log_timestep > args.log_interval) or (last_log_timestep == 0):
            logger.log_stat("episode", episode_num, runner.steps)
            logger.print_recent_stats()
            last_log_timestep = runner.steps

    runner.close_env()
    logger.info("Finish training!")


def run(run_sacred, config, log):
    # 配置参数的检查和调整
    config_sanity_check_and_adjust(config, log)
    args = SimpleNamespace(**config)

    logger = Logger(log)
    logger.info("Experiment Parameters:")
    experiment_params = pprint.pformat(config, indent=4, width=1)
    logger.info(experiment_params)

    # 设置 tensorboard
    if config["use_tensorboard"]:
        tb_log_dir = os.path.join(dirname(dirname(abspath(__file__))), "results/tb_logs/" + config["unique_token"])
        logger.setup_tb(tb_log_dir)
    # 开启 sacred
    logger.setup_sacred(run_sacred)

    # 启动采集和训练
    run_sequential(args=args, logger=logger)
