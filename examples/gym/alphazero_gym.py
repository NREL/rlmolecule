from abc import abstractmethod

import numpy as np
import gym


class AlphaZeroGymEnv(gym.Wrapper):
    """Simple wrapper class for a gym env to run with alphazero.  For 
    convenience you can either pass an env or a name that is available through 
    the standard gym env maker."""

    def __init__(self, env: gym.Env = None, name: str = None):
        if env is None:
            env = gym.envs.make(name)
        super().__init__(env)

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        raise NotImplementedError
