"""

"""
from math import ceil

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import sys

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

import command_line_tools

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

        from examples.gym.molecule_gym.molecule_graph_problem import MoleculeGraphProblem
        from rlmolecule.graph_gym.graph_gym_env import GraphGymEnv
        from rlmolecule.molecule.builder.builder import MoleculeBuilder

        return GraphGymEnv(MoleculeGraphProblem(MoleculeBuilder()))


    from rlmolecule.graph_gym.graph_gym_model import GraphGymModel


    class ThisModel(GraphGymModel):
        def __init__(self,
                     obs_space,
                     action_space,
                     num_outputs,
                     model_config,
                     name,
                     **kwargs):

            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            from rlmolecule.molecule.policy.model import policy_model

            # super(ThisModel, self).__init__(
            #     obs_space, action_space, num_outputs, model_config, name,
            #     MoleculeModel(policy_model()))
            super(ThisModel, self).__init__(
                obs_space, action_space, num_outputs, model_config, name,
                policy_model())


    register_env('molecule_graph_problem', make_env)

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

    num_workers = 11
    rollout_fragment_length = 12
    config = dict(
        {
            'local_dir': '../log',
            'env': 'molecule_graph_problem',
            'model': {
                'custom_model': 'molecule_graph_problem_model',
            },
            'num_gpus': .8,
            'num_gpus_per_worker': 0,
            'num_workers': num_workers,
            # 'num_gpus': 0,
            # 'num_gpus_per_worker': 0,
            # 'num_workers': 0,
            # 'num_gpus': 0,
            # 'num_gpus_per_worker': 0,
            # 'num_workers': num_workers,
            'framework': 'tf2',
            'eager_tracing': True,
            # 'framework': 'tf1',
            # 'rollout_fragment_length': int(8),
            # 'train_batch_size': int(16),
            # 'sgd_minibatch_size': 8,
            'rollout_fragment_length': rollout_fragment_length,
            'train_batch_size': int(ceil(64 / num_workers) * rollout_fragment_length),
            'sgd_minibatch_size': 64,  # 32 * num_workers / 2,
            'kl_target': 1e-4,
            'kl_coeff': 1e-5,
            'use_gae': False,
            # 'lambda': 0.99,
            'vf_clip_param': 1.0,
            "batch_mode": 'complete_episodes',  # '"truncate_episodes",
        },
        **cfg)

    config = command_line_tools.parse_config_from_args(sys.argv[1:], config)

    # stop = {
    #     'training_iteration': args['stop_iters'],
    #     'timesteps_total': args['stop_timesteps'],
    #     'episode_reward_mean': args['stop_reward'],
    # }

    local_dir = config['local_dir']
    del config['local_dir']

    results = tune.run(args['run'], config=config, verbose=3, local_dir=local_dir)

    if args['as_test']:
        check_learning_achieved(results, args['stop_reward'])

    ray.shutdown()
