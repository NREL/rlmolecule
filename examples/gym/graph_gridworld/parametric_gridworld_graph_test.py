"""Example of handling variable length and/or parametric action spaces.
This is a toy example of the action-embedding based approach for handling large
discrete action spaces (potentially infinite in size), similar to this:
    https://neuro.cs.ut.ee/the-use-of-embeddings-in-openai-five/
This currently works with RLlib's policy gradient style algorithms
(e.g., PG, PPO, IMPALA, A2C) and also DQN.
Note that since the model outputs now include "-inf" tf.float32.min
values, not all algorithm options are supported at the moment. For example,
algorithms might crash if they don't properly ignore the -inf action scores.
Working configurations are given below.
"""
import tensorflow as tf
from ray.rllib.models.tf import FullyConnectedNetwork

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

if __name__ == "__main__":
    # args = parser.parse_args()
    ray.init()

    args = {
        'run': 'PPO',
        'as_test': True,
        'stop_iters': int(1e3),
        'stop_timesteps': int(1e6),
        'stop_reward': 100.0,
    }


    def make_env(_):
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        from examples.gym.gridworld_env import make_doorway_grid, GridWorldEnv
        from rlmolecule.graph_gym.graph_gym_env import GraphGymEnv
        from examples.gym.graph_gridworld.gridworld_graph_problem import GridWorldGraphProblem

        return GraphGymEnv(GridWorldGraphProblem(GridWorldEnv(make_doorway_grid())))
        # return ParametricGridWorldEnv(GridWorldEnv(make_doorway_grid()))


    example_env = make_env(None)

    from examples.gym.parametric_gridworld_actions_model import ParametricGridworldActionsModel

    from rlmolecule.graph_gym.graph_gym_model import GraphGymModel

    class ModelForThisGridworld(GraphGymModel):
        def __init__(self,
                     obs_space,
                     action_space,
                     num_outputs,
                     model_config,
                     name,
                     # true_obs_shape=(4,),
                     # action_embed_size=2,
                     **kw):

            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            example_env = make_env(None)
            per_action_model = FullyConnectedNetwork(
                example_env.observation_space, action_space, 1,
                model_config, name + '_per_action_model')
            super(ModelForThisGridworld, self).__init__(
                obs_space, action_space, num_outputs, model_config, name,
                per_action_model, **kw)


    register_env('parametric_gridworld', make_env)

    ModelCatalog.register_custom_model('parametric_gridworld_model', ModelForThisGridworld)

    if args['run'] == 'DQN':
        cfg = {
            # TODO(ekl) we need to set these to prevent the masked values
            # from being further processed in DistributionalQModel, which
            # would mess up the masking. It is possible to support these if we
            # defined a custom DistributionalQModel that is aware of masking.
            'hiddens': [],
            'dueling': False,
        }
    else:
        cfg = {}

    config = dict(
        {
            'env': 'parametric_gridworld',
            'model': {
                'custom_model': 'parametric_gridworld_model',
            },
            'num_gpus': 0.2,
            'num_gpus_per_worker': 0.1,
            'num_workers': 4,
            'framework': 'tf2',
            'rollout_fragment_length': int(1e2),
            'train_batch_size': int(1e3),
        },
        **cfg)

    # stop = {
    #     'training_iteration': args['stop_iters'],
    #     'timesteps_total': args['stop_timesteps'],
    #     'episode_reward_mean': args['stop_reward'],
    # }

    results = tune.run(args['run'], config=config, verbose=3, local_dir='../../log')

    if args['as_test']:
        check_learning_achieved(results, args['stop_reward'])

    ray.shutdown()
