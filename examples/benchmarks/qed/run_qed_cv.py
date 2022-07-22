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
from rlmolecule.molecule_model import MoleculeModel
from rlmolecule.policy.preprocessor import load_preprocessor
from sklearn.model_selection import ParameterGrid

tf1, tf, tfv = try_import_tf()
num_gpus = len(tf.config.list_physical_devices("GPU"))
print(f"{num_gpus = }")

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=int, default=0, help="index for the grid search")


param_grid = {
    "lr": [1e-4, 1e-3, 1e-2],
    "entropy_coeff": [0.001, 0.01, 0.05],
    "train_batch_size": [2000, 4000, 8000],
    "num_sgd_iter": [1, 5, 10],
    "sgd_minibatch_size": [64, 128, 256],
}


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

    config = ParameterGrid(param_grid)[args.i]
    print(f"running entry {args.i}")
    print(config)

    custom_model = MoleculeModel

    tune.run(
        "PPO",
        config=dict(
            config,
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
                "gamma": 1.0,
                "num_workers": 33,
            },
        ),
        local_dir=Path("/scratch", os.environ["USER"], "ray_results"),
    )

    ray.shutdown()
