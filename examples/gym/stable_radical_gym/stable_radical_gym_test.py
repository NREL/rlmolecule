"""

"""

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

from examples.gym.stable_radical_gym.stable_radical_graph_problem import StableRadicalGraphProblem
from examples.gym.stable_radical_gym.stable_radical_model import StableRadicalModel
from rlmolecule.graph_gym.graph_gym_env import GraphGymEnv
from rlmolecule.graph_gym.graph_gym_model import GraphGymModel
from rlmolecule.molecule.builder.builder import MoleculeBuilder

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
        return GraphGymEnv(StableRadicalGraphProblem(MoleculeBuilder()))


    class ThisModel(GraphGymModel):
        def __init__(self,
                     obs_space,
                     action_space,
                     num_outputs,
                     model_config,
                     name,
                     **kwargs):
            per_action_model = StableRadicalModel(make_env(None))
            super(ThisModel, self).__init__(
                obs_space, action_space, num_outputs, model_config, name,
                per_action_model,
                **kwargs)


    register_env('stable_radical_graph_problem', make_env)

    ModelCatalog.register_custom_model('stable_radical_graph_problem_model', ThisModel)

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
            'env': 'stable_radical_graph_problem',
            'model': {
                'custom_model': 'stable_radical_graph_problem_model',
            },
            'num_gpus': 0.0,
            'num_gpus_per_worker': 0.0,
            'num_workers': 6,
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

    results = tune.run(args['run'], config=config, verbose=3)

    if args['as_test']:
        check_learning_achieved(results, args['stop_reward'])

    ray.shutdown()
