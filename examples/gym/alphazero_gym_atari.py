from abc import abstractmethod

import numpy as np
import gym

class AlphaZeroAtariGymEnv(gym.Wrapper):
    """Gym wrapper class for a generic Atari gym env to run with alphazero.
    For convenience you can either pass an env or a name that is available through 
    the standard gym env maker."""

    def __init__(self, env: gym.Env = None, name: str = None, history_length: int = 4):
        self.env = env if env is not None else gym.envs.make(name)
        self.history_length = history_length

        self.state = None
        super().__init__(self.env)
    
    def reset(self):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        self.frame = self.env.reset()

        # For the initial state, we stack the first frame four times
        self.state = np.repeat(self.frame, self.history_length, axis=2)

        return self.state

    def step(self, action, render_mode=None):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
            render_mode: None doesn't render anything, 'human' renders the screen in a new window, 'rgb_array' returns an np.array with rgb values
        Returns:
            reward: The reward for taking that action
            terminal: Whether the game has ended
            new_frame: The raw new frame as a result of that action
            If render_mode is set to 'rgb_array' this also returns the rendered rgb_array
        """
        new_frame, reward, terminal, info = self.env.step(action)

        self.state = np.append(self.state[:, :, 1:], new_frame, axis=2)

        if render_mode == 'rgb_array':
            return new_frame, reward, terminal, {}, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()

        return new_frame, reward, terminal, {}
