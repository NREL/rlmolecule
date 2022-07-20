import os
from pathlib import Path

import ray
import rdkit
from graphenv.graph_env import GraphEnv
from ray import tune
from ray.rllib.utils.framework import try_import_tf
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.redox import RedoxState
from rlmolecule.molecule_model import MoleculeModel
from rlmolecule.policy.preprocessor import load_preprocessor

tf1, tf, tfv = try_import_tf()
num_gpus = len(tf.config.list_physical_devices("GPU"))
print(f"{num_gpus = }")

stability_model="/projects/rlmolecule/pstjohn/models/20210214_radical_stability_new_data/"
redox_model="/projects/rlmolecule/pstjohn/models/20210214_redox_new_data/"
bde_model="/projects/rlmolecule/pstjohn/models/20210216_bde_new_nfp/"

max_atoms = 40

redoxstate = RedoxState(
    rdkit.Chem.MolFromSmiles("C"),
    builder=MoleculeBuilder(max_atoms=max_atoms, cache=True, gdb_filter=False),
    smiles="C",
    max_num_actions=32,
    warn=False,
    prune_terminal_states=True,
    stability_model= stability_model,
    redox_model= redox_model,
    bde_model= bde_model,
)

print(redoxstate)

print('tune1')

if __name__ == "__main__":

    custom_model = MoleculeModel
    
    print('tune')
    
    tune.run(
        "PPO",
        config=dict(
            **{
                "env": GraphEnv,
                "env_config": {
                    "state": redoxstate,
                    "max_num_children": redoxstate.max_num_actions,
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
                "lr": 0.001,
                "entropy_coeff": 0.001,
                "num_sgd_iter": 5,
                "train_batch_size": 4000,
            },
        ),
        local_dir=Path("/scratch", os.environ["USER"], "ray_results_molecule"),
    )

    ray.shutdown()
