import base64
from copy import deepcopy
import pickle
from typing import Sequence


import numpy as np
import gym

from rlmolecule.tree_search.graph_search_state import GraphSearchState

from alphazero_gym import AlphaZeroGymEnv


class GymEnvState(GraphSearchState):
    """Gyn env state implementation that maps the gym API to the GraphSearchState
    interface.  Should work generically for any gym env."""

    def __init__(self, 
                 env: AlphaZeroGymEnv,
                 step_reward: float,
                 cumulative_reward: float,
                 done: bool) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.env = deepcopy(env)
        self.step_reward = step_reward
        self.cumulative_reward = cumulative_reward
        self.done = done

    def __repr__(self) -> str:
        return self.env.get_obs().__repr__()

    def equals(self, other: any) -> bool:
        return type(self) == type(other) and \
               np.all(np.isclose(self.env.get_obs(), other.env.get_obs()))   # legit?

    def hash(self) -> int:
        return hash(self.__repr__())

    def get_next_actions(self) -> Sequence[GraphSearchState]:
        next_actions = []
        if not self.done:
            for action in range(self.env.action_space.n):
                env_copy = deepcopy(self.env)
                _, step_rew, done, _ = env_copy.step(action)
                cumulative_rew = self.cumulative_reward + step_rew
                next_actions.append(
                    GymEnvState(env_copy, step_rew, cumulative_rew, done))
        return next_actions


class AtariGymEnvState(GymEnvState, GraphSearchState):
    def __init__(self, 
                 env: AlphaZeroGymEnv,
                 step_reward: float,
                 cumulative_reward: float,
                 done: bool) -> None:
        super().__init__(env, step_reward, cumulative_reward, done)

    def serialize(self) -> str:
        self_data = (
            self.step_reward,
            self.cumulative_reward,
            self.done,
            self.env.clone_full_state())
        return base64.b64encode(pickle.dumps(self_data)).decode('utf-8')

    def deserialize(self, data: str) -> GraphSearchState:
        data = pickle.loads(base64.b64decode(data))
        step_reward = data[0]
        cumulative_reward = data[1]
        done = data[2]
        self.env.restore_full_state(data[3])
        print(step_reward, cumulative_reward, done)
        return AtariGymEnvState(self.env, step_reward, cumulative_reward, done)


if __name__ == "__main__":

    import gym
    from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, LazyFrames

    class PongEnv(AlphaZeroGymEnv, gym.ObservationWrapper):
        def __init__(self, **kwargs):
            env = gym.envs.make("PongNoFrameskip-v4")
            env = GrayScaleObservation(env) # Turns RGB image to gray scale
            env = ResizeObservation(env, shape=84) # resizes image on a square with side length == shape
            env = FrameStack(env, num_stack=4) # collect num_stack number of frames and feed them to policy network
            super().__init__(env, **kwargs)
        
        def observation(self, obs) -> np.ndarray:
            return 2 * (np.array(LazyFrames(list(obs), self.lz4_compress))/255 - 0.5)

        def get_obs(self) -> np.ndarray:
            return self.observation(self.frames)
    
    env = PongEnv()
    env.reset()
    cum_rew = 0.
    for _ in range(100):
        _, rew, _, _ = env.step(env.action_space.sample())
        cum_rew += rew
        s = AtariGymEnvState(env, rew, cum_rew, False)
    print(s.step_reward, s.cumulative_reward, s.done)
    data = s.serialize()
    print(data)

    env2 = PongEnv()
    env2.reset()
    s2 = AtariGymEnvState(env2, 0, 0, False)
    env2 = s2.deserialize(data)

    # print(env2.__dict__)
    # print(env.__dict__)
