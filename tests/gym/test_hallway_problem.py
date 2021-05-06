from typing import Dict, Tuple

import numpy as np
import pytest

import tensorflow as tf
from tensorflow.keras import layers

from rlmolecule.gym.gym_state import GymEnvState
from rlmolecule.tree_search.reward import LinearBoundedRewardFactory
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.gym.gym_problem import GymProblem
from rlmolecule.alphazero.alphazero import AlphaZero
from rlmolecule.mcts.mcts import MCTS

from tests.gym.test_gym_state import make_hallway


def hallway_policy(obs_dim: int,
                   hidden_layers: int = 1,
                   hidden_dim: int = 16,
                   activation: str = "relu",
                   input_dtype: str = "float64") -> tf.keras.Model:

    # Position input
    input_dtype = tf.dtypes.as_dtype(input_dtype)
    obs = layers.Input(shape=[obs_dim, ], dtype=input_dtype, name="obs")

    # Dense layers and concatenation
    x = layers.Dense(hidden_dim, activation=activation)(obs)
    for _ in range(hidden_layers-1):
        x = layers.Dense(hidden_dim, activation=activation)(x)
    x = layers.BatchNormalization()(x)

    value_logit = layers.Dense(1, name="value")(x)
    pi_logit = layers.Dense(1, name="prior")(x)

    return tf.keras.Model([obs], [value_logit, pi_logit], name="policy_model")


@pytest.fixture
def problem(request):

    policy = request.param
    env = make_hallway()

    class Problem(GymProblem):
        def __init__(self, env):
            super().__init__(reward_class=LinearBoundedRewardFactory(), env=env)

        if policy == "tf":
            def policy_model(self) -> "tf.keras.Model":
                return hallway_policy(
                            obs_dim = self.env.observation_space.shape[0],
                            hidden_layers = 3,
                            hidden_dim = 16,
                            input_dtype="int64")  # make sure your input dtypes match!

        elif policy == "random":
            def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
                random_state = np.random.RandomState()
                children = parent.children
                priors = random_state.dirichlet(np.ones(len(children)))

                return (
                    random_state.random(),
                    {vertex: prior for vertex, prior in zip(children, priors)}
                )
        else:
            raise ValueError("Invalid policy type '{}'".format(policy))  

        def get_policy_inputs(self, state: GymEnvState) -> dict:
            return {"obs": self.env.get_obs()}
    
    return Problem(env)
    

@pytest.fixture
def solver(request):
    name = request.param
    if name == 'MCTS':
        return MCTS
    if name == 'AlphaZero':
        return AlphaZero
    raise ValueError('Unknown problem type "{}"'.format(name))


def setup_game(solver, problem):
    game = solver(problem)
    root = game._get_root()
    return game, root


@pytest.mark.parametrize("solver,problem",
                        [("MCTS", "random"),
                         ("AlphaZero", "random"),
                         ("AlphaZero", "tf")], indirect=True)
class TestHallwayProblem:

    def test_step_reward(self, solver, problem):
        game, root = setup_game(solver, problem)
        game._expand(root)
        step_reward = -1./problem.env.size
        assert np.isclose(problem.reward_wrapper(root.children[0]).raw_reward, step_reward)
        assert np.isclose(problem.reward_wrapper(root.children[1]).raw_reward, step_reward)

    # def test_game_reward(self, solver, problem):
    #     game, root = setup_game(solver, problem)
    #     history, reward = game.run(num_mcts_samples=5)

    # def test_worst_reward(self, solver, problem):
    #     game, root = setup_game(solver, problem)
    #     game._expand(root)


