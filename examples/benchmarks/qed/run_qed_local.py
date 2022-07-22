import ray
import rdkit.Chem
from graphenv.graph_env import GraphEnv
from ray.rllib.agents import ppo
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.qed import QEDState
from rlmolecule.molecule_model import MoleculeModel
from rlmolecule.policy.preprocessor import load_preprocessor

max_atoms = 40

ray.init()

qed_state = QEDState(
    rdkit.Chem.MolFromSmiles("C"),
    builder=MoleculeBuilder(max_atoms=max_atoms, cache=False, gdb_filter=False),
    smiles="C",
    max_num_actions=32,
    warn=False,
    prune_terminal_states=True,
)

# assert qed_state._using_ray
# assert qed_state.builder._using_ray


config = {
    "env": GraphEnv,
    "env_config": {
        "state": qed_state,
        "max_num_children": qed_state.max_num_actions,
    },
    "log_level": "INFO",
    "model": {
        "custom_model": MoleculeModel,
        "custom_model_config": {
            "preprocessor": load_preprocessor(),
            "features": 32,
            "num_messages": 1,
        },
    },
    "eager_tracing": False,
    "framework": "tf2",
    "num_workers": 1,
}

trainer = ppo.PPOTrainer(config=config)
trainer.train()

ray.shutdown()
