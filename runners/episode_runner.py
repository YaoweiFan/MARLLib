import os
from os.path import dirname
import numpy as np
import imageio
import pandas as pd

from MARLLib.envs import ENV
from MARLLib.utils.buffer import EpisodeBatch
from MARLLib.utils.vec_normalize import VecNormalize


class EpisodeRunner:

    def __init__(self,
                 logger,
                 device,
                 batch_size_run,
                 checkpoint_path,
                 evaluate_args,
                 env,
                 env_args):

        self.logger = logger
        self.batch_size = batch_size_run
        assert self.batch_size == 1, "bath_size_run must be 1 when evaluate only!"
        self.device = device
        # 创建环境，获取环境信息
        self.env = ENV[env](**env_args)
        env_info = self.get_env_info()
        self.episode_limit = env_info["episode_limit"]

        self.returns = []
        self.stats = {}
        self.episode_step = None
        self.steps = None
        self.normalizer = None
        self.controller = None
        self.scheme = None
        self.groups = None
        self.preprocess = None
        self.controller = None
        self.batch = None

        self.checkpoint_path = os.path.join(dirname(dirname(dirname(__file__))), checkpoint_path)

        # record video
        self.video_record = evaluate_args["video_record"]
        self.skip_frame = evaluate_args["skip_frame"]
        self.video_save_dir = evaluate_args["video_save_path"]
        if self.video_record:
            os.makedirs(os.path.join(self.checkpoint_path, self.video_save_dir), exist_ok=True)
        # record path
        self.path_record = evaluate_args["path_record"]
        self.path_save_dir = evaluate_args["path_save_path"]
        if self.path_record:
            os.makedirs(os.path.join(self.checkpoint_path, self.path_save_dir), exist_ok=True)
        # record ft
        self.ft_record = evaluate_args["ft_record"]
        self.ft_save_dir = evaluate_args["ft_save_path"]
        if self.ft_record:
            os.makedirs(os.path.join(self.checkpoint_path, self.ft_save_dir), exist_ok=True)
        # record state
        self.state_record = evaluate_args["state_record"]
        self.state_save_dir = evaluate_args["state_save_path"]
        if self.state_record:
            os.makedirs(os.path.join(self.checkpoint_path, self.state_save_dir), exist_ok=True)

        self.video_save_path = None
        self.path_save_path = None
        self.ft_save_path = None
        self.state_save_path = None

    def setup(self, scheme, groups, preprocess, controller):
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.controller = controller
        self.controller = controller

    def set_path_name(self, index):
        self.video_save_path = os.path.join(self.checkpoint_path, self.video_save_dir, "{}.avi".format(index))
        self.path_save_path = os.path.join(self.checkpoint_path, self.path_save_dir, "{}.csv".format(index))
        self.ft_save_path = os.path.join(self.checkpoint_path, self.ft_save_dir, "{}.csv".format(index))
        self.state_save_path = os.path.join(self.checkpoint_path, self.state_save_dir, "{}.csv".format(index))

    def set_steps(self, steps):
        self.steps = steps

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = EpisodeBatch(self.scheme, self.groups, self.batch_size, self.episode_limit + 1,
                                  preprocess=self.preprocess, device=self.device)
        self.env.reset()
        self.episode_step = 0

    def load_normalizer(self, path):
        self.normalizer = VecNormalize.load(path)

    def run(self):
        self.reset()

        terminated = False
        episode_return = 0
        self.controller.init_hidden(batch_size=self.batch_size)

        # 记录视频
        writer = imageio.get_writer(self.video_save_path, fps=20)
        frame_count = 0
        # 记录路径点
        left_points = []
        right_points = []
        # 记录路径点姿态
        left_eef_quaternion = []
        right_eef_quaternion = []
        # 记录 action
        left_acts = []
        right_acts = []
        # 记录 ft
        left_ft_only = []
        right_ft_only = []
        # 记录 state
        left_joint_pos = []
        right_joint_pos = []
        left_joint_vel = []
        right_joint_vel = []
        left_eef_pos = []
        right_eef_pos = []
        left_quaternion = []
        right_quaternion = []
        left_ft = []
        right_ft = []
        left_peg_pos = []
        right_peg_pos = []
        left_peg_to_hole = []
        right_peg_to_hole = []

        while not terminated:
            pre_transition_data = {
                "state": [self.normalizer.normalize_state(self.env.get_state(), test_mode=True)],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.normalizer.normalize_obs(self.env.get_obs(), test_mode=True)]
            }

            self.batch.update(pre_transition_data, ts=self.episode_step)
            actions = self.controller.select_actions(self.batch, self.episode_step, self.steps, test_mode=True)

            reward, terminated, info = self.env.step(actions.reshape(-1))
            self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated and not info["timeout"],)],
            }

            self.batch.update(post_transition_data, ts=self.episode_step)

            self.episode_step += 1

            if self.video_record:
                # dump a frame from every k frames
                frame_count += 1
                if frame_count % self.skip_frame == 0:
                    writer.append_data(self.env.get_frame())

            if self.path_record:
                # 记录路径点
                left_points.append(self.env.obs["robot0_eef_pos"])
                right_points.append(self.env.obs["robot1_eef_pos"])
                # 记录路径点姿态
                left_eef_quaternion.append(self.env.obs["robot0_eef_quat"])
                right_eef_quaternion.append(self.env.obs["robot1_eef_quat"])
                # 记录 action
                left_acts.append(actions[0][0])
                right_acts.append(actions[0][1])

            if self.ft_record:
                left_ft_only.append(self.env.obs["robot0_ft"])
                right_ft_only.append(self.env.obs["robot1_ft"])

            if self.state_record:
                left_joint_pos.append(self.env.obs["robot0_joint_pos"])
                right_joint_pos.append(self.env.obs["robot1_joint_pos"])
                left_joint_vel.append(self.env.obs["robot0_joint_vel"])
                right_joint_vel.append(self.env.obs["robot1_joint_vel"])
                left_eef_pos.append(self.env.obs["robot0_eef_pos"])
                right_eef_pos.append(self.env.obs["robot1_eef_pos"])
                left_quaternion.append(self.env.obs["robot0_eef_quat"])
                right_quaternion.append(self.env.obs["robot1_eef_quat"])
                left_ft.append(self.env.obs["robot0_ft"])
                right_ft.append(self.env.obs["robot1_ft"])
                left_peg_pos.append(self.env.obs["robot0_peg_pos"])
                right_peg_pos.append(self.env.obs["robot1_peg_pos"])
                left_peg_to_hole.append(self.env.obs["robot0_peg_to_hole"])
                right_peg_to_hole.append(self.env.obs["robot1_peg_to_hole"])

        # 结束 video 记录
        if self.video_record:
            writer.close()

        # 结束 path 记录
        if self.path_record:
            left_points = np.array(left_points)
            right_points = np.array(right_points)
            left_x, left_y, left_z = left_points[:, 0], left_points[:, 1], left_points[:, 2]
            right_x, right_y, right_z = right_points[:, 0], right_points[:, 1], right_points[:, 2]

            left_eef_quaternion = np.array(left_eef_quaternion)
            right_eef_quaternion = np.array(right_eef_quaternion)
            left_eef_quaternion_x, left_eef_quaternion_y, left_eef_quaternion_z, left_eef_quaternion_w = \
                left_eef_quaternion[:, 0], left_eef_quaternion[:, 1], \
                left_eef_quaternion[:, 2], left_eef_quaternion[:, 3]
            right_eef_quaternion_x, right_eef_quaternion_y, right_eef_quaternion_z, right_eef_quaternion_w = \
                right_eef_quaternion[:, 0], right_eef_quaternion[:, 1], \
                right_eef_quaternion[:, 2], right_eef_quaternion[:, 3]
            
            left_acts = np.array(left_acts)
            right_acts = np.array(right_acts)

            dataframe = pd.DataFrame({'left_x': left_x, 'left_y': left_y, 'left_z': left_z,
                                      'right_x': right_x, 'right_y': right_y, 'right_z': right_z,
                                      'left_eef_quaternion_x': left_eef_quaternion_x,
                                      'left_eef_quaternion_y': left_eef_quaternion_y,
                                      'left_eef_quaternion_z': left_eef_quaternion_z,
                                      'left_eef_quaternion_w': left_eef_quaternion_w,
                                      'right_eef_quaternion_x': right_eef_quaternion_x,
                                      'right_eef_quaternion_y': right_eef_quaternion_y,
                                      'right_eef_quaternion_z': right_eef_quaternion_z,
                                      'right_eef_quaternion_w': right_eef_quaternion_w,
                                      'left_acts': left_acts, 'right_acts': right_acts
                                      })
            dataframe.to_csv(self.path_save_path)

        # 结束 ft 记录
        if self.ft_record:
            left_ft_only = np.array(left_ft_only)
            right_ft_only = np.array(right_ft_only)
            left_fx, left_fy, left_fz, left_tx, left_ty, left_tz = \
                left_ft_only[:, 0], left_ft_only[:, 1], left_ft_only[:, 2], \
                left_ft_only[:, 3], left_ft_only[:, 4], left_ft_only[:, 5]
            right_fx, right_fy, right_fz, right_tx, right_ty, right_tz = \
                right_ft_only[:, 0], right_ft_only[:, 1], right_ft_only[:, 2], \
                right_ft_only[:, 3], right_ft_only[:, 4], right_ft_only[:, 5]
            dataframe = pd.DataFrame({'left_fx': left_fx, 'left_fy': left_fy, 'left_fz': left_fz,
                                      'left_tx': left_tx, 'left_ty': left_ty, 'left_tz': left_tz,
                                      'right_fx': right_fx, 'right_fy': right_fy, 'right_fz': right_fz,
                                      'right_tx': right_tx, 'right_ty': right_ty, 'right_tz': right_tz})
            dataframe.to_csv(self.ft_save_path)

        # 结束 state 记录
        if self.state_record:
            # left_joint_pos = np.array(left_joint_pos)
            # left_joint_pos_1, left_joint_pos_2, left_joint_pos_3,
            # left_joint_pos_4, left_joint_pos_5, left_joint_pos_6, left_joint_pos_7 =
            # left_joint_pos[:, 0], left_joint_pos[:, 1], left_joint_pos[:, 2],
            # left_joint_pos[:, 3], left_joint_pos[:, 4], left_joint_pos[:, 5], left_joint_pos[:, 6]

            # right_joint_pos = np.array(right_joint_pos)
            # right_joint_pos_1, right_joint_pos_2, right_joint_pos_3,
            # right_joint_pos_4, right_joint_pos_5, right_joint_pos_6, right_joint_pos_7 =
            # right_joint_pos[:, 0], right_joint_pos[:, 1], right_joint_pos[:, 2],
            # right_joint_pos[:, 3], right_joint_pos[:, 4], right_joint_pos[:, 5], right_joint_pos[:, 6]

            # left_joint_vel = np.array(left_joint_vel)
            # left_joint_vel_1, left_joint_vel_2, left_joint_vel_3,
            # left_joint_vel_4, left_joint_vel_5, left_joint_vel_6, left_joint_vel_7 =
            # left_joint_vel[:, 0], left_joint_vel[:, 1], left_joint_vel[:, 2],
            # left_joint_vel[:, 3], left_joint_vel[:, 4], left_joint_vel[:, 5], left_joint_vel[:, 6]

            # right_joint_vel = np.array(right_joint_vel)
            # right_joint_vel_1, right_joint_vel_2, right_joint_vel_3,
            # right_joint_vel_4, right_joint_vel_5, right_joint_vel_6, right_joint_vel_7 =
            # right_joint_vel[:, 0], right_joint_vel[:, 1], right_joint_vel[:, 2],
            # right_joint_vel[:, 3], right_joint_vel[:, 4], right_joint_vel[:, 5], right_joint_vel[:, 6]

            left_eef_pos = np.array(left_eef_pos)
            left_eef_pos_1, left_eef_pos_2, left_eef_pos_3 = \
                left_eef_pos[:, 0], left_eef_pos[:, 1], left_eef_pos[:, 2]
            right_eef_pos = np.array(right_eef_pos)
            right_eef_pos_1, right_eef_pos_2, right_eef_pos_3 = \
                right_eef_pos[:, 0], right_eef_pos[:, 1], right_eef_pos[:, 2]
            left_quaternion = np.array(left_quaternion)
            left_quaternion_x, left_quaternion_y, left_quaternion_z, left_quaternion_w = \
                left_quaternion[:, 0], left_quaternion[:, 1], left_quaternion[:, 2], left_quaternion[:, 3]
            right_quaternion = np.array(right_quaternion)
            right_quaternion_x, right_quaternion_y, right_quaternion_z, right_quaternion_w = \
                right_quaternion[:, 0], right_quaternion[:, 1], right_quaternion[:, 2], right_quaternion[:, 3]
            # left_ft = np.array(left_ft)
            # left_ft_1, left_ft_2, left_ft_3, left_ft_4, left_ft_5, left_ft_6 = \
            #     left_ft[:, 0], left_ft[:, 1], left_ft[:, 2], left_ft[:, 3], left_ft[:, 4], left_ft[:, 5]
            # right_ft = np.array(right_ft)
            # right_ft_1, right_ft_2, right_ft_3, right_ft_4, right_ft_5, right_ft_6 = \
            #     right_ft[:, 0], right_ft[:, 1], right_ft[:, 2], right_ft[:, 3], right_ft[:, 4], right_ft[:, 5]
            left_peg_pos = np.array(left_peg_pos)
            left_peg_pos_x, left_peg_pos_y, left_peg_pos_z = \
                left_peg_pos[:, 0], left_peg_pos[:, 1], left_peg_pos[:, 2]
            right_peg_pos = np.array(right_peg_pos)
            right_peg_pos_x, right_peg_pos_y, right_peg_pos_z = \
                right_peg_pos[:, 0], right_peg_pos[:, 1], right_peg_pos[:, 2]
            left_peg_to_hole = np.array(left_peg_to_hole)
            left_peg_to_hole_x, left_peg_to_hole_y, left_peg_to_hole_z = \
                left_peg_to_hole[:, 0], left_peg_to_hole[:, 1], left_peg_to_hole[:, 2]
            right_peg_to_hole = np.array(right_peg_to_hole)
            right_peg_to_hole_x, right_peg_to_hole_y, right_peg_to_hole_z = \
                right_peg_to_hole[:, 0], right_peg_to_hole[:, 1], right_peg_to_hole[:, 2]

            dataframe = pd.DataFrame({
             'left_x': left_eef_pos_1, 'left_y': left_eef_pos_2, 'left_z': left_eef_pos_3,
             'right_x': right_eef_pos_1, 'right_y': right_eef_pos_2, 'right_z': right_eef_pos_3,
             'left_quaternion_x': left_quaternion_x, 'left_quaternion_y': left_quaternion_y,
             'left_quaternion_z': left_quaternion_z, 'left_quaternion_w': left_quaternion_w,
             'right_quaternion_x': right_quaternion_x, 'right_quaternion_y': right_quaternion_y,
             'right_quaternion_z': right_quaternion_z, 'right_quaternion_w': right_quaternion_w,
             'left_peg_pos_x': left_peg_pos_x, 'left_peg_pos_y': left_peg_pos_y, 'left_peg_pos_z': left_peg_pos_z,
             'right_peg_pos_x': right_peg_pos_x, 'right_peg_pos_y': right_peg_pos_y, 'right_peg_pos_z': right_peg_pos_z,
             'left_pth_x': left_peg_to_hole_x, 'left_pth_y': left_peg_to_hole_y, 'left_pth_z': left_peg_to_hole_z,
             'right_pth_x': right_peg_to_hole_x, 'right_pth_y': right_peg_to_hole_y, 'right_pth_z': right_peg_to_hole_z,
             'left_acts': left_acts, 'right_acts': right_acts
            })
            dataframe.to_csv(self.state_save_path)

        last_data = {
            "state": [self.normalizer.normalize_state(self.env.get_state(), test_mode=True)],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.normalizer.normalize_obs(self.env.get_obs(), test_mode=True)]
        }
        self.batch.update(last_data, ts=self.episode_step)

        # Select actions in the last stored state
        actions = self.controller.select_actions(self.batch, self.episode_step, self.steps, test_mode=True)
        self.batch.update({"actions": actions}, ts=self.episode_step)

        self.returns.append(episode_return)

    def log(self):
        self.logger.info("return_mean: {}".format(np.mean(self.returns)))
        self.logger.info("return_std: {}".format(np.std(self.returns)))

    def _print_obs(self):
        print("-------------------------------------------------------")
        print("panda1_obs:")
        print(self.env.get_obs()[0][0:3])
        print(self.env.get_obs()[0][3:7])
        print(self.env.get_obs()[0][7:13])
        print(self.env.get_obs()[0][13:16])
        print(self.env.get_obs()[0][16:19])
        print("panda2_obs:")
        print(self.env.get_obs()[1][0:3])
        print(self.env.get_obs()[1][3:7])
        print(self.env.get_obs()[1][7:13])
        print(self.env.get_obs()[1][13:16])
        print(self.env.get_obs()[1][16:19])
        print("-------------------------------------------------------")
