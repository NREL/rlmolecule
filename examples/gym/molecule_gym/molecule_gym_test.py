"""

"""

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
        from examples.gym.molecule_gym.molecule_graph_problem import MoleculeGraphProblem
        from rlmolecule.graph_gym.graph_gym_env import GraphGymEnv
        from rlmolecule.molecule.builder.builder import MoleculeBuilder
        return GraphGymEnv(MoleculeGraphProblem(MoleculeBuilder()))


    # def make_parametric_gridworld(_):
    #     from examples.gym.gridworld_env import GridWorldEnv, make_doorway_grid
    #     from examples.gym.parametric_gridworld_env import ParametricGridWorldEnv
    #     print('make_parametric_gridworld()')
    #     return ParametricGridWorldEnv(GridWorldEnv(make_doorway_grid()))

    from rlmolecule.graph_gym.graph_gym_model import GraphGymModel


    class ThisModel(GraphGymModel):
        def __init__(self,
                     obs_space,
                     action_space,
                     num_outputs,
                     model_config,
                     name,
                     **kwargs):
            from examples.gym.molecule_gym.molecule_model import MoleculeModel
            from rlmolecule.molecule.policy.model import policy_model
            # inner_action_model = policy_model()
            # per_action_model = MoleculeModel(inner_action_model)
            # super(ThisModel, self).__init__(
            #     obs_space, action_space, num_outputs, model_config, name,
            #     per_action_model,
            #     **kwargs)
            super(ThisModel, self).__init__(
                obs_space, action_space, num_outputs, model_config, name,
                policy_model,
                **kwargs)


    register_env('molecule_graph_problem', make_env)
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

    num_workers = 3
    config = dict(
        {
            'env': 'molecule_graph_problem',
            'model': {
                'custom_model': 'molecule_graph_problem_model',
            },
            'num_gpus': .1,
            'num_gpus_per_worker': 0.1,
            'num_workers': 1,
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
            'rollout_fragment_length': int(16),
            'train_batch_size': int(128),
            'sgd_minibatch_size': 32,
            "batch_mode": "truncate_episodes",
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
