from abc import abstractmethod

import numpy as np
import gym

#from examples.gym.atari_wrapper import GameWrapper
from examples.gym.frame_preprocessing import process_frame

class AlphaZeroAtariGymEnv(gym.Wrapper):
    """Gym wrapper class for Atari gym env to run with alphazero. The wrapper takes 
    a regular Atari observation of dimension 210x160x3 and transforms it into a
    84x84x1, with the new image being in greyscale.
    For convenience you can either pass an env or a name that is available through 
    the standard gym env maker."""

    def __init__(self, env: gym.Env = None, name: str = None, history_length: int = 4):
        self.env = env if env is not None else gym.envs.make(name)
        self.history_length = 4

        self.state = None
        self.last_lives = 0
        super().__init__(self.env)
    
    def reset(self):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        self.frame = self.env.reset()
        self.last_lives = 0

        # For the initial state, we stack the first frame four times
        self.state = np.repeat(process_frame(self.frame), self.history_length, axis=2)

        return self.state

    def step(self, action, render_mode=None):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
            render_mode: None doesn't render anything, 'human' renders the screen in a new window, 'rgb_array' returns an np.array with rgb values
        Returns:
            processed_frame: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
            life_lost: Whether a life has been lost
            new_frame: The raw new frame as a result of that action
            If render_mode is set to 'rgb_array' this also returns the rendered rgb_array
        """
        new_frame, reward, terminal, info = self.env.step(action)

        # In the commonly ignored 'info' or 'meta' data returned by env.step
        # we can get information such as the number of lives the agent has.

        # We use this here to find out when the agent loses a life, and
        # if so, we set life_lost to True.

        # We use life_lost to force the agent to start the game
        # and not sit around doing nothing.
        if info['ale.lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        if render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()

        return processed_frame, reward, terminal, life_lost

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        raise NotImplementedError
