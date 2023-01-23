import pytest
import rdkit
from graphenv.graph_env import GraphEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rlmolecule.builder import MoleculeBuilder
from rlmolecule.examples.qed import QEDState
from rlmolecule.molecule_model import MoleculeModel
from rlmolecule.molecule_state import MoleculeData
from rlmolecule.policy.preprocessor import load_preprocessor


@pytest.fixture
def ppo_config():
    ppo_config = (
            PPOConfig()
            .training(lr=1e-3, 
                      train_batch_size=20, 
                      sgd_minibatch_size=2,
                      shuffle_sequences=True,
                      num_sgd_iter=1)
            .resources(num_gpus=0)
            .framework("tf2")
            .rollouts(num_rollout_workers=1,  #parallelism
                      rollout_fragment_length=5)
            .debugging(log_level="DEBUG")
            )

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

    ppo_config.environment(env='QEDGraphEnv')
    ppo_config.training(model={"custom_model": "MoleculeModel", 
                               "custom_model_config": {
                                   "preprocessor": load_preprocessor(), 
                                   "features": 32, 
                                   "num_messages": 1, 
                                   },
                               },
                        )
    trainer = ppo_config.build()
    trainer.train()
