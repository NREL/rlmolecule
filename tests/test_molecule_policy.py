import numpy as np
import pytest
from graphenv.graph_env import GraphEnv
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.models.preprocessors import get_preprocessor
from rlmolecule.molecule_model import MoleculeModel
from rlmolecule.molecule_state import MoleculeState


@pytest.fixture
def single_layer_model(qed_root: MoleculeState, molecule_env: GraphEnv):
    return MoleculeModel(
        molecule_env.observation_space,
        molecule_env.action_space,
        num_outputs=1,
        model_config={},
        name="test_molecule_model",
        preprocessor=qed_root.preprocessor,
        features=8,
        num_messages=1,
    )


@pytest.fixture
def molecule_env(qed_root: MoleculeState):
    return GraphEnv({"state": qed_root, "max_num_children": qed_root.max_num_actions})


def test_policy_model(molecule_env, single_layer_model):

    observation, reward, terminal, info = molecule_env.step(0)

    preprocessor = get_preprocessor(molecule_env.observation_space)
    obs = preprocessor(molecule_env.observation_space).transform(observation)
    obs = _unpack_obs(obs[np.newaxis, :], molecule_env.observation_space)

    action_weights, state = single_layer_model.forward({"obs": obs}, None, None)

    # assert np.isfinite(value_logit[observation["action_mask"]]).all()
    # assert np.isfinite(prior_logit[observation["action_mask"]]).all()
