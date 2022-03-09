import pytest
import rdkit.Chem
from graphenv.graph_env import GraphEnv
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rdkit.Chem.QED import qed
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.molecule_model import MoleculeModel
from rlmolecule.molecule_state import MoleculeState


class QEDState(MoleculeState):
    @property
    def reward(self) -> float:
        if self.forced_terminal:
            return qed(self.molecule)
        else:
            return 0.0


class MoleculeEnv(GraphEnv):
    def __init__(self, config: EnvContext, *args, **kwargs) -> None:
        mol = rdkit.Chem.MolFromSmiles(config["initial_smiles"])
        builder = MoleculeBuilder(**config["builder"])
        state = QEDState(mol, builder=builder, smiles=config["initial_smiles"])
        super().__init__(state, *args, **kwargs)


@pytest.fixture
def ppo_config():

    config = {
        "num_gpus": 0,
        "num_workers": 1,  # parallelism
        "framework": "tf2",
        "eager_tracing": False,
        "eager_max_retraces": 20,
        "rollout_fragment_length": 5,
        "train_batch_size": 20,
        "sgd_minibatch_size": 2,
        "shuffle_sequences": True,
        "num_sgd_iter": 1,
        "lr": 1e-3,
    }

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)

    return ppo_config


def test_ppo(ray_init, ppo_config):

    ModelCatalog.register_custom_model("MoleculeModel", MoleculeModel)
    register_env("MoleculeEnv", lambda config: MoleculeEnv(config))

    config = {
        "env": "MoleculeEnv",
        "env_config": {"initial_smiles": "C", "builder": {"max_atoms": 5}},
        "model": {
            "custom_model": "MoleculeModel",
            "custom_model_config": {
                "features": 32,
                "num_messages": 1,
            },
        },
    }
    ppo_config.update(config)
    trainer = ppo.PPOTrainer(config=ppo_config)
    trainer.train()
