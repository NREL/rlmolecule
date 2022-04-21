import numpy as np
import pytest
from graphenv.graph_env import GraphEnv
from rlmolecule.molecule_state import MoleculeState
from rlmolecule.policy.model import policy_model


@pytest.fixture
def single_layer_model(qed_root: MoleculeState):
    return policy_model(qed_root.preprocessor, features=8, num_heads=2, num_messages=1)


@pytest.fixture
def molecule_env(qed_root: MoleculeState):
    return GraphEnv({"state": qed_root, "max_num_children": qed_root.max_num_actions})


def test_policy_model(molecule_env, single_layer_model):
    observation, reward, terminal, info = molecule_env.step(0)
    value_logit, prior_logit = single_layer_model.predict_on_batch(
        observation["vertex_observations"]
    )
    assert np.isfinite(value_logit[observation["action_mask"]]).all()
    assert np.isfinite(prior_logit[observation["action_mask"]]).all()
