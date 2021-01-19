from typing import List, Optional


class HallwayConfig:
    def __init__(self, 
                 size: int = 5,
                 terminal_reward: float = 10,
                 step_reward: float = -1):
        self.size = size
        self.terminal_reward = terminal_reward
        self.step_reward = step_reward
