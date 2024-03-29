import os
from os.path import dirname
import enum
from absl import logging  # information print
from robosuite.controllers import load_controller_config
import robosuite as suite
import numpy as np
import pandas as pd
from robosuite.environments.robot_env import RobotEnv

from .multi_agent_env import MultiAgentEnv


class Direction(enum.IntEnum):
    FRONT = 0
    BEHIND = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5


class DualArmRodEnv(MultiAgentEnv):
    """Dual-arm assemble environment for decentralised multi-agent coordination scenarios."""

    def __init__(
            self,
            seed,
            reward_shaping,
            reward_mimic,
            reward_success,
            reward_defeat,
            reward_scale,
            episode_limit,
            obs_timestep_number,
            debug,

            joint_pos_size,
            joint_vel_size,
            eef_pos_size,
            eef_quat_size,
            ft_size,
            peg_pos_size,
            peg_to_hole_size,
            robot_state_size,
            object_state_size,
            n_agents,
            n_actions,
            obs_choose,
            trajectory_data_path,
            ms_xl,
            ms_xh,
            ms_yl,
            ms_yh,
            ms_zl,
            ms_zh,

            has_renderer,
            has_offscreen_renderer,
            ignore_done,
            use_camera_obs,
            control_freq,
            camera_name,
            camera_heights,
            camera_widths,
    ):
        super().__init__()

        # MuJoCo
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.ignore_done = ignore_done
        self.use_camera_obs = use_camera_obs
        self.control_freq = control_freq
        self.camera_name = camera_name
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths

        # Observation and state
        self.joint_pos_size = joint_pos_size
        self.joint_vel_size = joint_vel_size
        self.eef_pos_size = eef_pos_size
        self.eef_quat_size = eef_quat_size
        self.ft_size = ft_size
        self.peg_pos_size = peg_pos_size
        self.peg_to_hole_size = peg_to_hole_size
        self.robot_state_size = robot_state_size
        self.object_state_size = object_state_size
        self.obs_choose = obs_choose
        self.obs = None

        # Action
        self.last_action = None
        self.n_actions = n_actions

        # Agents
        self.n_agents = n_agents

        # Range
        self.ms_xl = ms_xl
        self.ms_xh = ms_xh
        self.ms_yl = ms_yl
        self.ms_yh = ms_yh
        self.ms_zl = ms_zl
        self.ms_zh = ms_zh

        # Rewards
        self.reward_shaping = reward_shaping
        self.reward_mimic = reward_mimic
        self.reward_success = reward_success
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale

        # Algorithm
        self.obs_timestep_number = obs_timestep_number
        self.episode_limit = episode_limit
        self.debug = debug
        self.seed = seed
        self.episode_count = 0
        self.episode_steps = 0
        self.total_steps = 0
        self.timeouts = 0
        self.assemble_success = 0
        self.assemble_game = 0

        # Mimic trajectory
        absl_path = os.path.join(dirname(dirname(dirname(__file__))), trajectory_data_path)
        trajectory_data = pd.read_csv(absl_path)
        self.robot0_trajectory = np.array([[trajectory_data.Px.array[i],
                                            trajectory_data.Py.array[i],
                                            trajectory_data.Pz.array[i]] for i in range(len(trajectory_data.Px.array))])

        self.robot1_trajectory = np.array([[trajectory_data.Qx.array[i],
                                            trajectory_data.Qy.array[i],
                                            trajectory_data.Qz.array[i]] for i in range(len(trajectory_data.Qx.array))])

    def _launch(self):
        """Launch the Dual-arm assemble environment."""
        options = {"env_name": "TwoArmRod", "env_configuration": "single-arm-parallel", "robots": ["Panda", "Panda"]}
        controller_name = "OSC_POSITION"
        options["controller_configs"] = load_controller_config(default_controller=controller_name)

        self.env = suite.make(
            **options,
            has_renderer=self.has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            ignore_done=self.ignore_done,
            use_camera_obs=self.use_camera_obs,
            control_freq=self.control_freq,
            horizon=self.episode_limit,
            reward_shaping=self.reward_shaping,
            reward_scale=self.reward_scale,
            camera_names=self.camera_name,
            camera_heights=self.camera_heights,
            camera_widths=self.camera_widths,
        )
        return True

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self.episode_steps = 0
        if self.episode_count == 0:
            self._launch()
        self.obs = self.env.reset()

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.debug:
            logging.debug("Started Episode {}".format(self.episode_count).center(60, "*"))

    def _mimic_reward(self):
        """Return mimic reward."""
        mimic_reward = 0

        for robot_id in range(self.n_agents):
            min_dis = 1000000000

            check_data = None
            pos_curr = None
            if robot_id == 0:
                pos_curr = self.obs["robot0_eef_pos"]
                check_data = self.robot0_trajectory[self.episode_steps:]
            elif robot_id == 1:
                pos_curr = self.obs["robot1_eef_pos"]
                check_data = self.robot1_trajectory[self.episode_steps:]

            for i in range(len(check_data)):
                tmp_dis = np.linalg.norm(check_data[i] - pos_curr)
                if tmp_dis < min_dis:
                    min_dis = tmp_dis

            mimic_reward -= min_dis

        return 10 * mimic_reward

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions_int = [int(a) for a in actions]
        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            sc_action = self.get_agent_action(a_id, action)
            sc_actions.append(sc_action)

        # Execute actions
        # sc_actions = np.array([np.array([0,0.2,0,-1]), np.array([0,0,0,-1])])
        # 步长设置  
        sac = np.array(sc_actions).reshape(-1)

        # # debug code
        # while True:
        #     self.env.step(sac)
        #     self.render()
        #     # sleep(1)

        self.obs, reward, terminated, info = self.env.step(sac)
        self.render()

        if self.reward_mimic:
            reward += self._mimic_reward()

        self.total_steps += 1
        self.episode_steps += 1

        if self.episode_steps >= self.episode_limit:
            self.timeouts += 1

        if terminated:
            self.episode_count += 1
            self.assemble_game += 1
            if info["success"]:
                self.assemble_success += 1
                reward += self.reward_success * (1 - self.reward_shaping)
            if info["defeat"]:
                reward += self.reward_defeat * (1 - self.reward_shaping)

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

           - joint_pos
           - joint_vel
           - eef_pos
           - eef_quat
           - ft
           - peg_pos
           - peg_to_hole

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. 

           NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """
        assert isinstance(self.env, RobotEnv)
        prefix = self.env.robots[agent_id].robot_model.naming_prefix

        if self.obs_choose == "noft":
            agent_obs = np.concatenate(
                (
                    self.obs[prefix + "eef_pos"],
                    self.obs[prefix + "eef_quat"],
                    self.obs[prefix + "peg_pos"],
                    self.obs[prefix + "peg_to_rod_top"],
                )
            )
        elif self.obs_choose == "noeef":
            agent_obs = np.concatenate(
                (
                    self.obs[prefix + "eef_ft"],
                    self.obs[prefix + "peg_pos"],
                    self.obs[prefix + "peg_to_rod_top"],
                )
            )
        elif self.obs_choose == "nopegpos":
            agent_obs = np.concatenate(
                (
                    self.obs[prefix + "eef_pos"],
                    self.obs[prefix + "eef_quat"],
                    self.obs[prefix + "eef_ft"],
                    self.obs[prefix + "peg_to_rod_top"],
                )
            )
        else:
            agent_obs = np.concatenate(
                (
                    # self.obs[prefix + "joint_pos"],
                    # self.obs[prefix + "joint_vel"],
                    self.obs[prefix + "eef_pos"],
                    self.obs[prefix + "eef_quat"],
                    # self.obs[prefix + "eef_ft"],
                    self.obs[prefix + "ft"],
                    self.obs[prefix + "peg_pos"],
                    self.obs[prefix + "peg_to_rod_top"],
                )
            )

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs, self.episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Obs Robot: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Joint position {}".format(self.obs[prefix + "joint_pos"]))
            logging.debug("Joint velocity {}".format(self.obs[prefix + "joint_vel"]))
            logging.debug("EEF position {}".format(self.obs[prefix + "eef_pos"]))
            logging.debug("EEF quaternion {}".format(self.obs[prefix + "eef_quat"]))
            logging.debug("Force and torque {}".format(self.obs[prefix + "ft"]))
            logging.debug("Peg position {}".format(self.obs[prefix + "peg_pos"]))
            logging.debug("Peg to hole {}".format(self.obs[prefix + "peg_to_hole"]))

        return agent_obs

    def get_state(self):
        """Returns the global state.
        This function assumes that self.obs is up-to-date.
        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.concatenate(
            (
                self.obs["robot0_robot-state"],
                self.obs["robot1_robot-state"],
                self.obs["object-state"],
            )
        )
        return state

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        if avail_actions[action] == 0:
            assert avail_actions[action] == 1, "Agent {} cannot perform action {}".format(a_id, action)

        if action == 0:
            # stop
            robo_act = np.array([0, 0, 0, 1])
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 1:
            # move front
            robo_act = np.array([1, 0, 0, 1])
            if self.debug:
                logging.debug("Agent {}: Move Front".format(a_id))

        elif action == 2:
            # move behind
            robo_act = np.array([-1, 0, 0, 1])
            if self.debug:
                logging.debug("Agent {}: Move Behind".format(a_id))

        elif action == 3:
            # move left
            robo_act = np.array([0, -1, 0, 1])
            if self.debug:
                logging.debug("Agent {}: Move Left".format(a_id))

        elif action == 4:
            # move right
            robo_act = np.array([0, 1, 0, 1])
            if self.debug:
                logging.debug("Agent {}: Move Right".format(a_id))

        elif action == 5:
            # move up
            robo_act = np.array([0, 0, 1, 1])
            if self.debug:
                logging.debug("Agent {}: Move Up".format(a_id))

        else:
            # move down
            robo_act = np.array([0, 0, -1, 1])
            if self.debug:
                logging.debug("Agent {}: Move Down".format(a_id))

        return robo_act

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        avail_actions = [0] * self.n_actions
        # stop should be allowed
        avail_actions[0] = 1

        # see if we can move
        if self.can_move(agent_id, Direction.FRONT):
            avail_actions[1] = 1
        if self.can_move(agent_id, Direction.BEHIND):
            avail_actions[2] = 1
        if self.can_move(agent_id, Direction.LEFT):
            avail_actions[3] = 1
        if self.can_move(agent_id, Direction.RIGHT):
            avail_actions[4] = 1
        if self.can_move(agent_id, Direction.UP):
            avail_actions[5] = 1
        if self.can_move(agent_id, Direction.DOWN):
            avail_actions[6] = 1

        return avail_actions

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def can_move(self, agent_id, direction):
        """Whether a robot can move in a given direction.
        This function assumes that self.obs is up-to-date.
        """
        assert isinstance(self.env, RobotEnv)
        prefix = self.env.robots[agent_id].robot_model.naming_prefix
        posx, posy, posz = self.obs[prefix + "eef_pos"]

        if direction == Direction.FRONT:
            return posx <= self.ms_xh

        if direction == Direction.BEHIND:
            return posx >= self.ms_xl

        if direction == Direction.RIGHT:
            return posy <= self.ms_yh

        if direction == Direction.LEFT:
            return posy >= self.ms_yl

        if direction == Direction.UP:
            return posz <= self.ms_zh

        if direction == Direction.DOWN:
            return posz >= self.ms_zl

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.robot_state_size * 2 + self.object_state_size

    def get_obs_size(self):
        """Returns the size of the observation."""
        if self.obs_choose == "noft":
            return self.eef_pos_size + self.eef_quat_size + self.peg_pos_size + self.peg_to_hole_size
        if self.obs_choose == "noeef":
            return self.ft_size + self.peg_pos_size + self.peg_to_hole_size
        if self.obs_choose == "nopegpos":
            return self.eef_pos_size + self.eef_quat_size + self.ft_size + self.peg_to_hole_size
        return self.eef_pos_size + self.eef_quat_size + self.ft_size + self.peg_pos_size + self.peg_to_hole_size

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def render(self):
        """Render"""
        if self.has_renderer:
            self.env.render()

    def close(self):
        """Close the environment"""
        self.env.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self.seed

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "assemble_success": self.assemble_success,
            "assemble_games": self.assemble_game,
            "win_rate": self.assemble_success / self.assemble_game,
            "timeouts": self.timeouts,
        }
        return stats

    def get_frame(self):
        return self.obs[self.camera_name + "_image"][::-1]
