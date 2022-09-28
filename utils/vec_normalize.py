import pickle
import numpy as np
from typing import Tuple


class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, batch_mean, batch_var, batch_count) -> None:
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class VecNormalize:
    """
    A moving average, normalizing wrapper for vectorized environment.
    has support for saving/loading moving average,
    """

    def __init__(
        self,
        obs_shape,
        state_shape,
        clip_obs,
        clip_state,
        epsilon,
        use_running_normalize,
    ):

        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.state_rms = RunningMeanStd(shape=state_shape)
        self.clip_obs = clip_obs
        self.clip_state = clip_state
        self.epsilon = epsilon
        self.use_running_normalize = use_running_normalize

    def normalize_obs(self, obs, test_mode):
        if not self.use_running_normalize:
            return obs
        obs_flattened = np.concatenate(obs)
        if not test_mode:
            self.obs_rms.update(obs_flattened, 0, 1)
        # Normalize observations
        obs_normalized = np.clip((obs_flattened - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                                 -self.clip_obs, self.clip_obs).astype(np.float32)
        return [obs_normalized[0:obs[0].shape[0]], obs_normalized[obs[0].shape[0]:]]

    def normalize_state(self, state, test_mode):
        if not self.use_running_normalize:
            return state
        if not test_mode:
            self.state_rms.update(state, 0, 1)
        # Normalize state
        state_normalized = np.clip((state - self.state_rms.mean) / np.sqrt(self.state_rms.var + self.epsilon),
                                   -self.clip_state, self.clip_state).astype(np.float32)
        return state_normalized

    @staticmethod
    def load(load_path: str):
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        return vec_normalize

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)
