import pytest
import rdkit
from graphenv.graph_env import GraphEnv
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.qed import QEDState
from rlmolecule.molecule_model import MoleculeModel
from rlmolecule.molecule_state import MoleculeData
from rlmolecule.policy.preprocessor import load_preprocessor


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
    def create_env(config):

        data = MoleculeData(
            MoleculeBuilder(max_atoms=5, cache=True),
            max_num_actions=20,
            prune_terminal_states=True,
        )

        qed_root = QEDState(
            rdkit.Chem.MolFromSmiles("C"),
            data=data,
            smiles="C",
        )

        return GraphEnv(
            {"state": qed_root, "max_num_children": qed_root.max_num_actions}
        )

    ModelCatalog.register_custom_model("MoleculeModel", MoleculeModel)

    register_env("QEDGraphEnv", lambda config: create_env(config))

    config = {
        "env": "QEDGraphEnv",
        "model": {
            "custom_model": "MoleculeModel",
            "custom_model_config": {
                "preprocessor": load_preprocessor(),
                "features": 32,
                "num_messages": 1,
            },
        },
    }
    ppo_config.update(config)
    trainer = ppo.PPOTrainer(config=ppo_config)
    trainer.train()
