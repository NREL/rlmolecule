import pytest

import numpy as np

from rlmolecule.gym.gym_state import GymEnvState
from rlmolecule.gym.alphazero_gym import AlphaZeroGymEnv

from tests.gym.gridworld_env import (GridWorldEnv, PLAYER_CHANNEL, GOAL_CHANNEL)
from tests.gym.hallway_env import HallwayEnv


def _raise_env_error(name):
    raise ValueError("Invalid env name '{}'".format(name))

# Both envs in the tests have a get_obs method
class TestGym(AlphaZeroGymEnv):
    def get_obs(self):
        return self.env.get_obs()

@pytest.fixture(params=["grid", "hallway"])
def env(request):
    name = request.param
    if name == "grid":
        size = 5
        grid = np.zeros((size, size, 3))
        grid[0, 0, PLAYER_CHANNEL] = 1
        grid[-1, -1, GOAL_CHANNEL] = 1
        env = TestGym(GridWorldEnv(grid=grid))
    elif name == "hallway":
        env = TestGym(HallwayEnv(size=4, max_steps=4))
    else:
        _raise_env_error(name)
    env.reset()
    return env


def test_parent_child_unequal(env):
    """No children are equal to parent."""
    state = GymEnvState(env, 0, 0., 0., False)
    for next_state in state.get_next_actions():
        assert not state.equals(next_state)
    
def test_reset_env_is_equal(env):
    state = GymEnvState(env, 0, 0., 0., False)
    _ = [env.step(env.action_space.sample()) for _ in range(3)]
    env.reset()
    assert state.equals(GymEnvState(env, 0, 0., 0., False))

def test_done(env):
    """Zero next actions when `done`"""
    state = GymEnvState(env, 0, 0., 0., True)
    assert len(state.get_next_actions()) == 0

def test_step_changes_state(env):
    """Step is part of state and distinguishes otherwise identical envs."""
    state1 = GymEnvState(env, 0, 0., 0., False)
    state2 = GymEnvState(env, 1, 0., 0., False)
    assert not state1.equals(state2)

def test_step_reward_does_not_change_equals(env):
    """Step reward does not affect the equals call."""
    state1 = GymEnvState(env, 0, 0., 0., False)
    state2 = GymEnvState(env, 0, 1., 0., False)
    assert state1.equals(state2)

def test_cumulative_reward_does_not_change_equals(env):
    """Cumulative reward does not effect equals call."""
    state1 = GymEnvState(env, 0, 0., 0., False)
    state2 = GymEnvState(env, 0, 0., 2., False)
    assert state1.equals(state2)

def test_done_does_not_change_equals(env):
    """Cumulative reward does not effect equals call."""
    state1 = GymEnvState(env, 0, 0., 0., False)
    state2 = GymEnvState(env, 0, 0., 0., True)
    assert state1.equals(state2)

