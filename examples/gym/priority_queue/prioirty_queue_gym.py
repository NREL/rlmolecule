import heapq
from typing import Tuple, Dict, Optional

import gym
import numpy as np

from rlmolecule.graph_gym.graph_problem import GraphProblem
from rlmolecule.tree_search.graph_search_state import GraphSearchState


class PriorityQueueGym(gym.Env):

    def __init__(self,
                 problem: GraphProblem,
                 ) -> None:
        super().__init__()
        self.problem: GraphProblem = problem
        self.action_space: gym.Space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float)
        self.observation_space: gym.Space = problem.observation_space

        self.state: Optional[GraphSearchState] = None
        self.new: [GraphSearchState] = []
        self.open: [(float, int, GraphSearchState)] = []
        self.closed: {GraphSearchState} = set()
        self.reset()

    def reset(self) -> {str: np.ndarray}:
        self.state: Optional[GraphSearchState] = self.problem.get_initial_state()
        self.new: [GraphSearchState] = []
        self.open: [(float, int, GraphSearchState)] = []
        self.closed: {GraphSearchState} = set()

        return self.make_observation()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        heapq.heappush(self.open, (action[0], len(self.closed), self.state))

        total_reward = 0.0
        while (len(self.new) <= 0) and (len(self.open) > 0):
            priority, count, parent = heapq.heappop(self.open)
            self.closed.add(parent)

            for v in parent.get_next_actions():
                reward, _, _ = self.problem.step(v)
                total_reward += reward
                if v not in self.closed:
                    self.new.append(v)

        is_terminal = len(self.new) == 0
        if not is_terminal:
            self.state = self.new.pop(-1)

        result = (self.make_observation(), total_reward, is_terminal, {})
        print(f'PriorityQueueGym: {total_reward}, {len(self.new)}, {len(self.open)} {len(self.closed)}')
        return result

    def make_observation(self) -> {str: any}:
        return self.problem.make_observation(self.state)
