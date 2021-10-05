"""

"""
# print('******************* import tensorflow as tf - 0')
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import os
import sys

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

import command_line_tools

THIS_DR = os.path.abspath(os.path.dirname(__file__))

# This param HARDCODED, also in policy.policy_model optional args bc currently
# things aren't set up to pass optional args to the policy constructor.
# IF you change it here, you must also a) ensure that pca-{size}.p file has
# been generated in the fingerprint_and_pca.ipynb notebook and b) change the
# input_dim default in policy_model to match.
PCA_EMBED_SIZE = 256


if __name__ == "__main__":
    # args = parser.parse_args()
    ray.init()

    args = {
        'run': 'PPO',
        'as_test': True,
        'stop_iters': int(1e3),
        'stop_timesteps': int(1e5),
        'stop_reward': 1.0,
    }


    def make_env(_):
        # print('begin make_env()')
        # print('******************* import tensorflow as tf - 1')
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        from examples.gym.mol_fp_gym.molecule_graph_problem import MoleculeGraphProblem
        from rlmolecule.graph_gym.graph_gym_env import GraphGymEnv
        from rlmolecule.molecule.builder.builder import MoleculeBuilder

        # Load the PCA model for embedding
        import pickle
        with open(os.path.join("pca-{}.p".format(PCA_EMBED_SIZE)), "rb") as f:
            pca = pickle.load(f)

        result = GraphGymEnv(
            MoleculeGraphProblem(
                MoleculeBuilder(max_atoms=6, min_atoms=1),
                pca
            )
        )
        # print('end make_env()')
        return result


    #from rlmolecule.graph_gym.graph_gym_model import GraphGymModel
    from examples.gym.parametric_gridworld_actions_model import ParametricGridworldActionsModel

    example_env = make_env(None)
    #print("ENV ACTION SPACE", example_env.action_space)
    #print("ENV OBS SPACE", example_env.observation_space)

    #class ThisModel(GraphGymModel):
    class ThisModel(ParametricGridworldActionsModel):
        def __init__(self,
                     obs_space,
                     action_space,
                     num_outputs,
                     model_config,
                     name,
                     **kwargs):
            # print('******************* import tensorflow as tf - 2')
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            # from examples.gym.mol_fp_gym.molecule_model import MoleculeModel
            # from examples.gym.mol_fp_gym.policy import policy_model

            super(ThisModel, self).__init__(
                obs_space, action_space, num_outputs, model_config, name,
                #policy_model,
                example_observation_space=example_env.observation_space,
                **kwargs)


    env_name = 'mol_fp_pca={}'.format(PCA_EMBED_SIZE)
    register_env(env_name, make_env)
    # register_env('parametric_gridworld', make_parametric_gridworld)

    ModelCatalog.register_custom_model('molecule_graph_problem_model', ThisModel)

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

    num_workers = 34
    rollout_fragment_length = 12
    config = dict(
        {
            'local_dir': '../log',
            'env': env_name,
            'model': {
                'custom_model': 'molecule_graph_problem_model',
            },
            'num_gpus': 1,
            'num_gpus_per_worker': 0,
            'num_workers': num_workers,
            "lr": tune.grid_search([1e-4]),
            'framework': 'tf2',
            'eager_tracing': False,
            "num_sgd_iter": 10,
            "entropy_coeff": tune.grid_search([0.0]),
            'rollout_fragment_length': rollout_fragment_length,
            'train_batch_size': num_workers*rollout_fragment_length,
            'sgd_minibatch_size': num_workers*rollout_fragment_length,
            "batch_mode": 'complete_episodes',  # '"truncate_episodes",
            "log_level": "WARN"
        },
        **cfg)

    config = command_line_tools.parse_config_from_args(sys.argv[1:], config)

    stop = {
        'training_iteration': args['stop_iters'],
        'timesteps_total': args['stop_timesteps'],
        'episode_reward_mean': args['stop_reward'],
    }

    local_dir = config['local_dir']
    del config['local_dir']

    results = tune.run(
        args['run'],
        stop=stop,
        num_samples=3,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=10,
        config=config,
        verbose=3,
        local_dir=local_dir)

    if args['as_test']:
        check_learning_achieved(results, args['stop_reward'])

    ray.shutdown()
