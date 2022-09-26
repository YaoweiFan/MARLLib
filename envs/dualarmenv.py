from smac.env.multiagentenv import MultiAgentEnv

import enum
from absl import logging # infomation print

import sys
sys.path.append('/home/fyw/Documents/projects/dual-arm-mimic')
from robosuite.controllers import load_controller_config
from robosuite.utils import *
import robosuite as suite
import numpy as np
import pandas as pd

from gym.spaces import Discrete

class Direction(enum.IntEnum):
    FRONT = 0
    BEHIND = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5

class DualArmEnv(MultiAgentEnv):
    """Dual-arm assemble environment for decentralised multi-agent coordination scenarios."""
    def __init__(
        self,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
        camera_name="frontview",
        camera_heights=512,
        camera_widths=512,
        obs_choose="all",
                
        seed=None,
        reward_shaping=False,
        reward_mimic=True,
        reward_success=200,
        reward_defeat=-200,
        reward_scale=3.0,
        reward_separate=False,
        debug=False,
        n_agents=2,
        episode_limit=100,
        replay_dir="",
        replay_prefix="",
        state_last_action=True,
        state_timestep_number=False,
        obs_timestep_number=False,
    ):
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        seed : int, optional
            Random seed used during game initialisation. This allows to
        reward_shaping : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_success : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for losing in an episode (default is -200).
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        """
        # MuJoCo environment params
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.ignore_done = ignore_done
        self.use_camera_obs = use_camera_obs
        self.control_freq = control_freq
        self.camera_name = camera_name
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.obs_choose = obs_choose
        print(self.obs_choose)

        # Observation and state
        self.joint_pos_size = 7
        self.joint_vel_size = 7
        self.eef_pos_size = 3
        self.eef_quat_size = 4
        self.ft_size = 6
        self.peg_pos_size = 3
        self.peg_to_hole_size = 3
        self.robot_state_size = 38
        self.object_state_size = 19

        # Action
        self.n_actions = 7

        # Rewards args
        self.reward_shaping = reward_shaping
        self.reward_mimic = reward_mimic
        self.reward_success = reward_success
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale
        self.reward_separate = reward_separate

        # Other
        self.debug = debug
        self._seed = seed
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix
        self.obs_timestep_number = obs_timestep_number

        # Assemble task params
        self.n_agents = n_agents
        self.episode_limit = episode_limit
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.ms_xl = -0.25
        self.ms_xh = 0.25
        self.ms_yl = -0.3
        self.ms_yh = 0.3
        self.ms_zl = 0
        self.ms_zh = 1.5
        self.timeouts = 0
        self.assemble_success = 0
        self.assemble_game = 0

        # rmappo use
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.n_agents):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append(self.get_obs_size())
            self.share_observation_space.append(self.get_share_obs_size())

        # mimic
        trajectory_data = pd.read_csv('data/trajectory.csv')
 
        self.robot0_trajectory = np.array([ [ trajectory_data.Px.array[i], 
                                              trajectory_data.Py.array[i], 
                                              trajectory_data.Pz.array[i] ] 
                                              for i in range(len(trajectory_data.Px.array)) ])

        self.robot1_trajectory = np.array([ [ trajectory_data.Qx.array[i], 
                                              trajectory_data.Qy.array[i], 
                                              trajectory_data.Qz.array[i] ] 
                                              for i in range(len(trajectory_data.Qx.array)) ])

    def _launch(self):
        """Launch the Dual-arm assemble environment."""
        options = {}
        options["env_name"] = "TwoArmAssemble"
        options["env_configuration"] = "single-arm-parallel"
        options["robots"] = []
        for i in range(2):
            options["robots"].append("Panda")
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
        self._episode_steps = 0
        if self._episode_count == 0:
            self._launch()
        self._obs = self.env.reset()

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))
        # 控制夹爪闭合
        for i in range(3):
            self.env.step(np.array([0,0,0,1,0,0,0,1]))
            self.render()
        # return self.get_obs(), self.get_state(), self.get_avail_actions()

    def _mimic_reward(self):
        """Return mimic reward."""
        mimic_reward = 0

        for robot_id in range(2):
            min_dis = 1000000000

            if robot_id == 0:
                pos_curr = self._obs["robot0_eef_pos"]
                check_data = self.robot0_trajectory[self._episode_steps:]
            elif robot_id == 1:
                pos_curr = self._obs["robot1_eef_pos"]
                check_data = self.robot1_trajectory[self._episode_steps:]

            for i in range(len(check_data)):
                tmp_dis = np.linalg.norm( check_data[i] - pos_curr )
                if tmp_dis < min_dis:
                    min_dis = tmp_dis

            mimic_reward -= min_dis

        return 10*mimic_reward

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

            
        self._obs, reward, terminated, info = self.env.step(sac)
        self.render()
        
        if self.reward_mimic:
            reward += self._mimic_reward()

        self._total_steps += 1
        self._episode_steps += 1

        if self._episode_steps >= self.episode_limit:
            self.timeouts += 1

        if terminated:
            self._episode_count += 1
            self.assemble_game += 1
            if info["success"] == True:
                self.assemble_success += 1
                reward += self.reward_success * (1 - self.reward_shaping)
            if info["defeat"] == True:
                reward += self.reward_defeat * (1 - self.reward_shaping)

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))

        if self.reward_separate:
            return self.get_obs(), self.get_share_obs(), [[reward]]*self.n_agents,\
                    [terminated]*self.n_agents, info, self.get_avail_actions()
        else:
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
        prefix = self.env.robots[agent_id].robot_model.naming_prefix

        if self.obs_choose == "noft":
            agent_obs = np.concatenate(
                (
                    self._obs[prefix + "eef_pos"],
                    self._obs[prefix + "eef_quat"],
                    self._obs[prefix + "peg_pos"],
                    self._obs[prefix + "peg_to_hole"],
                )
            ) 
        elif self.obs_choose == "noeef":
            agent_obs = np.concatenate(
                (
                    self._obs[prefix + "eef_ft"],
                    self._obs[prefix + "peg_pos"],
                    self._obs[prefix + "peg_to_hole"],
                )
            )
        elif self.obs_choose == "nopegpos":
            agent_obs = np.concatenate(
                (
                    self._obs[prefix + "eef_pos"],
                    self._obs[prefix + "eef_quat"],
                    self._obs[prefix + "eef_ft"],
                    self._obs[prefix + "peg_to_hole"],
                )
            )
        else:
            agent_obs = np.concatenate(
                (
                    # self._obs[prefix + "joint_pos"],
                    # self._obs[prefix + "joint_vel"],
                    self._obs[prefix + "eef_pos"],
                    self._obs[prefix + "eef_quat"],
                    # self._obs[prefix + "eef_ft"],
                    np.array([0, 0, 0, 0, 0, 0]),
                    # self._obs[prefix + "ft"],
                    self._obs[prefix + "peg_pos"],
                    self._obs[prefix + "peg_to_hole"],
                )
            )

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Obs Robot: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Joint position {}".format(self._obs[prefix + "joint_pos"]))
            logging.debug("Joint velocity {}".format(self._obs[prefix + "joint_vel"]))
            logging.debug("EEF position {}".format(self._obs[prefix + "eef_pos"]))
            logging.debug("EEF quaternion {}".format(self._obs[prefix + "eef_quat"]))
            logging.debug("Force and torque {}".format(self._obs[prefix + "ft"]))
            logging.debug("Peg position {}".format(self._obs[prefix + "peg_pos"]))
            logging.debug("Peg to hole {}".format(self._obs[prefix + "peg_to_hole"]))

        return agent_obs

    def get_state(self):
        """Returns the global state.
        This function assumes that self._obs is up-to-date.
        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.concatenate(
            (
                self._obs["robot0_robot-state"],
                self._obs["robot1_robot-state"],
                self._obs["object-state"],
            )
        )
        return state

    # rmappo use
    def get_share_obs(self):
        share_obs = [self.get_share_obs_agent(i) for i in range(self.n_agents)]
        return share_obs

    # rmappo use
    def get_share_obs_agent(self, agent_id):
        prefix = self.env.robots[agent_id].robot_model.naming_prefix

        agent_share_obs = np.concatenate(
            (
                self._obs[prefix + "joint_pos"],
                self._obs[prefix + "joint_vel"],
                self._obs[prefix + "eef_pos"],
                self._obs[prefix + "eef_quat"],
                self._obs[prefix + "ft"],
                self._obs["object-state"],
            )
        )

        if self.obs_timestep_number:
            agent_share_obs = np.append(agent_share_obs,
                                  self._episode_steps / self.episode_limit)

        return agent_share_obs

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert avail_actions[action] == 1, \
                "Agent {} cannot perform action {}".format(a_id, action)

        if action == 0:
            # stop
            robo_act = np.array([0,0,0,1])
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 1:
            # move front
            robo_act = np.array([1,0,0,1])
            if self.debug:
                logging.debug("Agent {}: Move Front".format(a_id))

        elif action == 2:
            # move behind
            robo_act = np.array([-1,0,0,1])
            if self.debug:
                logging.debug("Agent {}: Move Behind".format(a_id))

        elif action == 3:
            # move left
            robo_act = np.array([0,-1,0,1])
            if self.debug:
                logging.debug("Agent {}: Move Left".format(a_id))

        elif action == 4:
            # move right
            robo_act = np.array([0,1,0,1])
            if self.debug:
                logging.debug("Agent {}: Move Right".format(a_id))

        elif action == 5:
            # move up
            robo_act = np.array([0,0,1,1])
            if self.debug:
                logging.debug("Agent {}: Move Up".format(a_id))

        else:
            # move down
            robo_act = np.array([0,0,-1,1])
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
        This function assumes that self._obs is up-to-date.
        """
        prefix = self.env.robots[agent_id].robot_model.naming_prefix
        posx, posy, posz = self._obs[prefix + "eef_pos"]

        m = 0.001 

        if direction == Direction.FRONT:
            x, y, z = posx + m, posy, posz
        elif direction == Direction.BEHIND:
            x, y, z = posx - m, posy, posz
        elif direction == Direction.RIGHT:
            x, y, z = posx, posy + m, posz
        elif direction == Direction.LEFT:
            x, y, z = posx, posy - m, posz
        elif direction == Direction.UP:
            x, y, z = posx, posy, posz + m
        else:
            x, y, z = posx, posy, posz - m

        if self.check_bounds(x, y, z):
            return True

        return False
    
    def check_bounds(self, x, y, z):
        """Whether a point is within the motion space."""
        return (self.ms_xl <= x <= self.ms_xh and self.ms_yl <= y <= self.ms_yh
                and self.ms_zl <= z <= self.ms_zh)

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.robot_state_size * 2 + self.object_state_size
    
    def get_obs_size(self):
        """Returns the size of the observation."""
        if self.obs_choose == "noft":
            return  self.eef_pos_size + self.eef_quat_size + self.peg_pos_size + self.peg_to_hole_size
        if self.obs_choose == "noeef":
            return  self.ft_size + self.peg_pos_size + self.peg_to_hole_size
        if self.obs_choose == "nopegpos":
            return  self.eef_pos_size + self.eef_quat_size + self.ft_size + self.peg_to_hole_size
        return  self.eef_pos_size + self.eef_quat_size + self.ft_size + self.peg_pos_size + self.peg_to_hole_size

    # rmappo use
    def get_share_obs_size(self):
        return  self.joint_pos_size + self.joint_vel_size + self.eef_pos_size +\
                self.eef_quat_size + self.ft_size + self.object_state_size

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
        return self._seed

    def save_replay(self):
        """Save a replay."""
        pass

    def get_stats(self):
        stats = {
            "assemble_success": self.assemble_success,
            "assemble_games": self.assemble_game,
            "win_rate": self.assemble_success / self.assemble_game,
            "timeouts": self.timeouts,
        }
        return stats

    def get_frame(self):
        return self._obs[self.camera_name + "_image"][::-1]