import os
from pathlib import Path

import ray
import rdkit
from graphenv.graph_env import GraphEnv
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import register_env
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.qed import QEDState
from rlmolecule.molecule_model import MoleculeModel
from rlmolecule.policy.preprocessor import load_preprocessor

tf1, tf, tfv = try_import_tf()
num_gpus = len(tf.config.list_physical_devices("GPU"))
print(f"{num_gpus = }")

ModelCatalog.register_custom_model("MoleculeModel", MoleculeModel)
register_env("GraphEnv", lambda config: GraphEnv(config))

cache_dir = Path(os.environ["LOCAL_SCRATCH"], "pstjohn")

qed_state = QEDState(
    rdkit.Chem.MolFromSmiles("C"),
    builder=MoleculeBuilder(max_atoms=25, cache_dir=cache_dir),
    smiles="C",
    max_num_actions=50,
)

assert qed_state.max_num_actions == 50

config = {
    "env": "GraphEnv",
    "env_config": {"state": qed_state, "max_num_children": qed_state.max_num_actions},
    "model": {
        "custom_model": "MoleculeModel",
        "custom_model_config": {
            "preprocessor": load_preprocessor(),
            "features": 32,
            "num_messages": 1,
        },
    },
    "num_gpus": 1 if num_gpus >= 1 else 0,
    "num_workers": 35,  # parallelism
    "framework": "tf2",
    "eager_tracing": True,
    "batch_mode": "complete_episodes",
    "train_batch_size": 25 * 35 * 10,
    "shuffle_sequences": True,
    "lr": 1e-3,
}

ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update(config)


if __name__ == "__main__":
    ray.init(dashboard_host="0.0.0.0")

    tune.run(
        "PPO",
        config=ppo_config,
        local_dir=Path("/scratch", os.environ["USER"], "ray_results"),
    )

    ray.shutdown()
