import argparse
import os
from pathlib import Path

import ray
import rdkit
from graphenv.graph_env import GraphEnv
from ray import tune
from ray.rllib.utils.framework import try_import_tf
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.qed import QEDState
from rlmolecule.molecule_model import MoleculeModel, MoleculeQModel
from rlmolecule.policy.preprocessor import load_preprocessor

tf1, tf, tfv = try_import_tf()
num_gpus = len(tf.config.list_physical_devices("GPU"))
print(f"{num_gpus = }")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="DQN", help="The RLlib-registered algorithm to use."
)


ray.init(dashboard_host="0.0.0.0")


max_atoms = 40

qed_state = QEDState(
    rdkit.Chem.MolFromSmiles("C"),
    builder=MoleculeBuilder(max_atoms=max_atoms, cache=True, gdb_filter=False),
    smiles="C",
    max_num_actions=32,
    warn=False,
    prune_terminal_states=True,
)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.run in ["DQN", "APEX"]:
        extra_config = {
            "hiddens": [],
            "dueling": False,
            "v_min": 0,
            "v_max": 1,
        }

        if args.run == "DQN":
            extra_config.update(
                {
                    "lr": 1e-2,
                    "num_workers": 1,
                    "exploration_config": {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 1.0,
                        "final_epsilon": 0.05,
                        "warmup_timesteps": 0,
                        "epsilon_timesteps": int(1e5),
                    },
                }
            )

        if args.run == "APEX":
            extra_config.update(
                {
                    "num_workers": 32,
                    "timesteps_per_iteration": 1000,
                    "min_time_s_per_reporting": 10,
                    "lr": 0.001,
                    "optimizer": {"num_replay_buffer_shards": 3},
                    "replay_buffer_config": {
                        "capacity": 20000,
                        "learning_starts": 100,
                    },
                    "target_network_update_freq": 500,
                    "exploration_config": {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 1.0,
                        "final_epsilon": 0.05,
                        "warmup_timesteps": 100,
                        "epsilon_timesteps": 1000,
                    },
                }
            )

        custom_model = MoleculeQModel

    else:
        extra_config = {"num_workers": 35}
        custom_model = MoleculeModel

        if args.run == "PPO":
            extra_config.update(
                {
                    "lr_schedule": [[0, 1e-2], [10_000, 1e-3], [50_000, 1e-4]],
                    "gamma": 1.0,  # finite horizon problem, we want total reward-to-go?
                    # "entropy_coeff": tune.grid_search([0.05]),
                    "entropy_coeff_schedule": [
                        [0, 0.05],
                        [50_000, 0.01],
                        [75_000, 0.001],
                    ],
                    "rollout_fragment_length": max_atoms + 1,
                    "num_sgd_iter": 10,
                    "sgd_minibatch_size": 128,
                }
            )

        if args.run == "IMPALA":
            extra_config.update({"vtrace_drop_last_ts": False})

    tune.run(
        args.run,
        config=dict(
            extra_config,
            **{
                "env": GraphEnv,
                "env_config": {
                    "state": qed_state,
                    "max_num_children": qed_state.max_num_actions,
                },
                "model": {
                    "custom_model": custom_model,
                    "custom_model_config": {
                        "preprocessor": load_preprocessor(),
                        "features": 32,
                        "num_messages": 1,
                    },
                },
                "num_gpus": 1 if num_gpus >= 1 else 0,
                "framework": "tf2",
                "eager_tracing": True,
                "batch_mode": "complete_episodes",
            },
        ),
        local_dir=Path("/scratch", os.environ["USER"], "ray_results"),
    )

    ray.shutdown()
