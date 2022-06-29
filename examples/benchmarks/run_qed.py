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

cache_dir = Path(os.environ["LOCAL_SCRATCH"], "pstjohn")

qed_state = QEDState(
    rdkit.Chem.MolFromSmiles("C"),
    builder=MoleculeBuilder(max_atoms=40, cache_dir=cache_dir),
    smiles="C",
    max_num_actions=32,
    warn=False,
)


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(dashboard_host="0.0.0.0")

    if args.run in ["DQN", "APEX"]:
        extra_config = {
            "hiddens": [],
            "dueling": False,
            "v_min": 0,
            "v_max": 1,
        }

        if args.run == "APEX":
            extra_config.update(
                {
                    "learning_starts": 5000,
                    "target_network_update_freq": 50000,
                    "timesteps_per_iteration": 2500,
                }
            )

        custom_model = MoleculeQModel

    else:
        extra_config = {}
        custom_model = MoleculeModel

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
                "num_workers": 30,
                "framework": "tf2",
                "eager_tracing": True,
                "batch_mode": "complete_episodes",
            },
        ),
        local_dir=Path("/scratch", os.environ["USER"], "ray_results"),
    )

    ray.shutdown()
