# from typing import Dict, Tuple

# import numpy as np
# import pytest

# import tensorflow as tf
# from tensorflow.keras import layers

# from rlmolecule.gym.gym_state import GymEnvState
# from rlmolecule.tree_search.reward import LinearBoundedRewardFactory
# from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
# from rlmolecule.gym.gym_problem import GymProblem
# from rlmolecule.alphazero.alphazero import AlphaZero
# from rlmolecule.mcts.mcts import MCTS

# from tests.gym.test_gym_state import make_hallway, make_gridworld


# def hallway_policy(obs_dim: int,
#                    hidden_layers: int = 1,
#                    hidden_dim: int = 16,
#                    activation: str = "relu",
#                    input_dtype: str = "float64") -> tf.keras.Model:

#     # Position input
#     input_dtype = tf.dtypes.as_dtype(input_dtype)
#     obs = layers.Input(shape=[obs_dim, ], dtype=input_dtype, name="obs")

#     # Dense layers and concatenation
#     x = layers.Dense(hidden_dim, activation=activation)(obs)
#     for _ in range(hidden_layers-1):
#         x = layers.Dense(hidden_dim, activation=activation)(x)
#     x = layers.BatchNormalization()(x)

#     value_logit = layers.Dense(1, name="value")(x)
#     pi_logit = layers.Dense(1, name="prior")(x)

#     return tf.keras.Model([obs], [value_logit, pi_logit], name="policy_model")


# def gridworld_policy(obs_dim: Tuple[int],
#                      hidden_layers: int = 1,
#                      conv_layers: int = 2,
#                      filters_dim: list = [32, 64],
#                      kernel_dim: list = [8, 4],
#                      strides_dim: list = [4, 3],
#                      hidden_dim: int = 256,
#                      activation: str = "relu") -> tf.keras.Model:

#     obs = layers.Input(shape=obs_dim, dtype=tf.float64, name="obs")
#     steps = layers.Input(shape=(1,), dtype=tf.float64, name="steps")
    
#     x = layers.Conv2D(
#         filters_dim[0], 
#         (kernel_dim[0],kernel_dim[0]), 
#         strides_dim[0], 
#         activation=activation)(obs)
#     for i in range(conv_layers-1):
#         x = layers.Conv2D(
#             filters_dim[i],
#             (kernel_dim[i],kernel_dim[i]),
#             strides_dim[i],
#             activation=activation)(x)
#     x = layers.Flatten()(x)

#     x = layers.Concatenate()((x, steps))

#     for _ in range(hidden_layers):
#         x = layers.Dense(hidden_dim, activation=activation)(x)
#     x = layers.BatchNormalization()(x)
    
#     value_logit = layers.Dense(1, name="value")(x)
#     pi_logit = layers.Dense(1, name="prior")(x)

#     return tf.keras.Model([obs, steps], [value_logit, pi_logit], name="policy_model")


# @pytest.fixture
# def grid_problem(request):

#     policy = request.param
#     env = make_gridworld()

#     class Problem(GymProblem):

#         def __init__(self, env):
#             super().__init__(reward_class=LinearBoundedRewardFactory(), env=env)

#         if policy == "tf":
#             def policy_model(self) -> "tf.keras.Model":
#                 obs_shape = self.env.reset().shape
#                 return gridworld_policy(obs_dim=obs_shape,
#                                         hidden_layers=2,
#                                         conv_layers=1,
#                                         filters_dim=[16],
#                                         kernel_dim=[2],
#                                         strides_dim=[1],
#                                         hidden_dim=64)

#         elif policy == "random":
#             def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
#                 random_state = np.random.RandomState()
#                 children = parent.children
#                 priors = random_state.dirichlet(np.ones(len(children)))

#                 return (
#                     random_state.random(),
#                     {vertex: prior for vertex, prior in zip(children, priors)})
#         else:
#             raise ValueError("Invalid policy type '{}'".format(policy))

#         def get_policy_inputs(self, state: GymEnvState) -> dict:
#             return {
#                 "obs": state.env.get_obs(),
#                 "steps": 0.*np.array([np.float64(self.env.episode_steps)])
#             }

#     return Problem(env)


# @pytest.fixture
# def hallway_problem(request):

#     policy = request.params
#     env = make_hallway()

#     class Problem(GymProblem):
#         def __init__(self, env):
#             super().__init__(reward_class=LinearBoundedRewardFactory(), env=env)

#         if policy == "tf":
#             def policy_model(self) -> "tf.keras.Model":
#                 return hallway_policy(
#                             obs_dim = self.env.observation_space.shape[0],
#                             hidden_layers = 3,
#                             hidden_dim = 16,
#                             input_dtype="int64")  # make sure your input dtypes match!

#         elif policy == "random":
#             def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):
#                 random_state = np.random.RandomState()
#                 children = parent.children
#                 priors = random_state.dirichlet(np.ones(len(children)))

#                 return (
#                     random_state.random(),
#                     {vertex: prior for vertex, prior in zip(children, priors)}
#                 )
#         else:
#             raise ValueError("Invalid policy type '{}'".format(policy))  

#         def get_policy_inputs(self, state: GymEnvState) -> dict:
#             return {"obs": self.env.get_obs()}
    
#     return Problem(env)
    

# @pytest.fixture
# def solver(request):
#     name = request.param
#     if name == 'MCTS':
#         return MCTS
#     if name == 'AlphaZero':
#         return AlphaZero
#     raise ValueError('Unknown problem type "{}"'.format(name))


# def setup_hallway_game(solver, hallway_problem):
#     game = solver(grid_problem)
#     root = game._get_root()
#     return game, root


# @pytest.mark.parametrize("solver,problem",
#                         [("MCTS", "random"),
#                          ("AlphaZero", "random"),
#                          ("AlphaZero", "tf")], indirect=True)
# class TestHallway:

#     def test_reward(self, solver, problem):
#         game, root = setup_hallway_game(solver, problem)
#         game._expand(root)
#         assert problem.reward_wrapper(root.children[-1]).raw_reward < 0.

    

