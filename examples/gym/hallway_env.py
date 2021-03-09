from typing import Tuple

import numpy as np

import gym


class HallwayEnv(gym.Env):

    def __init__(self, size: int, max_steps: int):
        
        self.size = size
        self.max_steps = max_steps

        self.position = 0
        self.episode_steps = 0

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.size, self.max_steps]),
            dtype=np.int)

    def reset(self) -> np.ndarray:
        self.position = 0
        self.episode_steps = 0
        return self.get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action == 0:
            self.position = max(0, self.position - 1)
        else:
            self.position += 1
        self.episode_steps += 1
        done = (self.position == self.size) or (self.episode_steps == self.max_steps)
        return self.get_obs(), self.get_reward(), done, {}

    def get_reward(self) -> float:
        return -(self.episode_steps + (self.size - self.position))

    def get_obs(self) -> np.ndarray:
        return np.array([self.position, self.episode_steps], dtype=int)

    

if __name__ == "__main__":
    env = HallwayEnv(16, 16)
    done = False
    _ = env.reset()
    while not done:
        action = env.action_space.sample()
        print("action:", action)
        obs, rew, done, _ = env.step(action)
        print("result:", obs, rew, done)
