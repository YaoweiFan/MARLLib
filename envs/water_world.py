import sys
sys.path.append("/home/fyw/Documents/projects/maddpg/MADRL")
sys.path.append("/home/fyw/Documents/projects/maddpg/MADRL/rltools")
sys.path.append("/home/fyw/Documents/projects/maddpg/MADRL/rltools/rllab")
import numpy as np

from madrl_environments.pursuit import MAWaterWorld_mod
from .multi_agent_env import MultiAgentEnv


class WaterWorld(MultiAgentEnv):
    """Dual-arm assemble environment for decentralised multi-agent coordination scenarios."""

    def __init__(
            self,
            n_pursuers,
            n_evaders,
            n_poison,
            obstacle_radius,
            food_reward,
            poison_reward,
            encounter_reward,
            n_coop,
            sensor_range,
            seed,
            obstacle_loc=None,
    ):
        super().__init__()

        self.world = MAWaterWorld_mod(n_pursuers=n_pursuers, n_evaders=n_evaders,
                                      n_poison=n_poison, obstacle_radius=obstacle_radius,
                                      food_reward=food_reward,
                                      poison_reward=poison_reward,
                                      encounter_reward=encounter_reward,
                                      n_coop=n_coop,
                                      sensor_range=sensor_range, obstacle_loc=obstacle_loc)

        self.n_agents = 2
        self.n_obs = 213
        self.n_states = 426
        self.n_actions = 2
        self.seed = 1234
        self.episode_limit = 1000

        # self.episode_steps = 0
        self.obs = None
        self.states = None

        self.world.seed(self.seed)

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        # self.episode_steps = 0
        self.obs = self.world.reset()
        self.states = np.concatenate(self.obs, axis=0)

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info.
        actions: np.ndarray (n_agents, action_dim)
        """
        self.obs, reward, done, _ = self.world.step(actions)
        self.states = np.concatenate(self.obs, axis=0)
        info = {"timeout": True}

        # self.episode_steps += 1
        # if self.episode_steps == self.episode_limit:
        #     done = True
        #     info["timeout"] = True

        return reward, done, info

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return self.obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        pass

    def get_state(self):
        """Returns the global state.
        This function assumes that self.obs is up-to-date.
        NOTE: This functon should not be used during decentralised execution.
        """
        return self.states

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.n_states

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.n_obs

    def get_avail_actions(self):
        # Continuous environment, forbid to use this method
        raise Exception("Continuous environment, forbid to use this method!")

    def get_avail_agent_actions(self, agent_id):
        # Continuous environment, forbid to use this method
        raise Exception("Continuous environment, forbid to use this method!")

    def get_total_actions(self):
        """Returns the dim of actions an agent could ever take."""
        return self.n_actions

    def render(self):
        """Render"""
        self.world.render()

    def close(self):
        """Close the environment"""
        self.world.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self.seed

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "action_dim": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {}
        return stats
