import os
from pathlib import Path

from typing import Dict


import ray
import rdkit
from graphenv.graph_env import GraphEnv
from ray import tune
from ray.rllib.utils.framework import try_import_tf

from ray.tune.registry import register_env
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.qed import QEDState
from rlmolecule.molecule_model import MoleculeModel
from rlmolecule.molecule_state import MoleculeData

from rlmolecule.policy.preprocessor import load_preprocessor

tf1, tf, tfv = try_import_tf()
num_gpus = len(tf.config.list_physical_devices("GPU"))
#print(f"{num_gpus = }")

output_directory = Path("/home/ray")
Path(output_directory, "qed").mkdir(exist_ok=True)

#ray.init(dashboard_host="0.0.0.0")
# Ray will be already launched in AWS Ray clusters
ray.init(address="auto")

max_atoms = 40


def create_env(config: Dict):
    """When not running in local_mode, there are often issues in allowing ray to copy
    `MoleculeState` to distribute the environment on worker nodes, since actor handles
    are copied and not initialized correctly.

    To solve this, it's best to delay `MoleculeState` (and the dataclass) initialization
    until needed on each ray worker through the `register_env` method.

    Here, we create and return an initialized `GraphEnv` object.
    """

    qed_data = MoleculeData(
        MoleculeBuilder(max_atoms=max_atoms, cache=True, gdb_filter=False),
        max_num_actions=32,
        prune_terminal_states=True,
        log_reward_filepath=Path(output_directory, "qed", "eagle_results.csv"),
    )
    qed_state = QEDState(rdkit.Chem.MolFromSmiles("C"), qed_data, smiles="C",)
    return GraphEnv({"state": qed_state, "max_num_children": qed_state.max_num_actions})


# This registers the above function with rllib, such that we can pass only "QEDGraphEnv"
# as our env object in `tune.run()`
register_env("QEDGraphEnv", lambda config: create_env(config))


if __name__ == "__main__":

    custom_model = MoleculeModel

    tune.run(
        "PPO",
        config=dict(
            **{

                "env": "QEDGraphEnv",
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
                "lr": 0.001,
                "entropy_coeff": 0.001,
                "num_sgd_iter": 5,
                "train_batch_size": 4000,
            },
        ),
        local_dir=Path(output_directory, "ray_results"))

    ray.shutdown()
