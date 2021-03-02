import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from rlmolecule.alphazero.alphazero import AlphaZero
from rlmolecule.molecule.molecule_config import MoleculeConfig
from rlmolecule.tree_search.reward import RankedRewardFactory, RawRewardFactory
from tests.qed_optimization_problem import QEDWithMoleculePolicy


@pytest.fixture()
def config():
    return MoleculeConfig(max_atoms=4,
                          min_atoms=1,
                          tryEmbedding=False,
                          sa_score_threshold=None,
                          stereoisomers=False)


@pytest.fixture(scope='function')
def game(request, engine, tmpdirname, config):
    """
    The scope here is function, so that the problem gets recreated for each test. Otherwise the trained policy
    network doesn't need to be loaded, since the trained model already exists in the problem.
    """

    name = request.param

    if name == 'raw':
        reward_class = RawRewardFactory()
        noise = True

    elif name == 'ranked':
        reward_class = RankedRewardFactory(
            reward_buffer_min_size=2,
            reward_buffer_max_size=4,
            run_id=name,
            engine=engine
        )
        noise = True

    elif name == 'nonoise':
        reward_class = RawRewardFactory()
        noise = False

    else:
        raise RuntimeError(f"{name} not found")

    problem = QEDWithMoleculePolicy(engine,
                                    config,
                                    features=8,
                                    num_heads=2,
                                    num_messages=1,
                                    run_id=name,
                                    min_buffer_size=0,
                                    reward_class=reward_class,
                                    policy_checkpoint_dir=tmpdirname)

    return AlphaZero(problem, dirichlet_noise=noise)


@pytest.mark.parametrize('game', ['raw', 'ranked', 'nonoise'], indirect=True)
class TestPolicyTraining:

    def test_reward_caching(self, game):
        root = game._get_root()

        game.problem.get_reward = MagicMock(return_value=(1, {}))

        reward1 = game.problem.reward_wrapper(root.state).raw_reward
        reward2 = game.problem.reward_wrapper(root.state).raw_reward

        assert reward1 == reward2
        assert game.problem.get_reward.call_count == 1

    def test_create_games(self, game):
        final_mols = []
        rewards = []

        for i in range(5):
            history, reward = game.run(num_mcts_samples=5)
            assert history[0][0].visit_count == 5

            from rlmolecule.sql.tables import GameStore
            stored_game = game.problem.session.query(GameStore).filter_by(id=str(game.problem.id)).one()
            assert stored_game.scaled_reward == reward.scaled_reward
            assert stored_game.raw_reward == reward.raw_reward

            final_mols += [history[-1][0]]
            rewards += [reward]

        # Make sure there's some diversity in the final molecules
        assert len(set(final_mols)) > 1
        assert len(set(rewards)) > 1

    def test_recent_games(self, game):
        problem = game.problem
        recent_games = list(problem.iter_recent_games())
        assert len(recent_games) == 5

    def test_policy_data(self, game):
        problem = game.problem
        data = problem._create_dataset()
        inputs, (rewards, visit_probs) = list(data.take(1))[0]
        assert inputs['atom'].shape[1] == visit_probs.shape[1] + 1
        assert inputs['atom'].shape[0] == problem.batch_size

        outputs = problem.batched_policy_model(inputs)
        assert outputs[0].shape.as_list() == rewards.shape.as_list()
        assert outputs[1].shape.as_list() == visit_probs.shape.as_list()

    def test_policy_model_masking(self, game):
        problem = game.problem
        data = problem._create_dataset()
        inputs, (rewards, visit_probs) = list(data.take(1))[0]

        inputs_for_batch_layer = tf.keras.Model(problem.batched_policy_model.inputs,
                                                problem.batched_policy_model.layers[-1].input)(inputs)

        input_mask = [inp._keras_mask for inp in inputs_for_batch_layer]

        outputs = problem.batched_policy_model(inputs)
        output_mask = problem.batched_policy_model.layers[-1].compute_mask(inputs_for_batch_layer, input_mask)


        assert output_mask[0].numpy().all(), 'no values should be masked'

        # make sure that the final output masks matches where there's no atoms.
        assert (output_mask[1].numpy() == ~(inputs['atom'] == 0).numpy().all(-1)[:, 1:]).all()

        # make sure the number of actions is consistent
        assert (input_mask[0].numpy().sum(1) - 1 == output_mask[1].numpy().sum(1)).all()

    def test_train_policy_model(self, game):
        problem = game.problem

        weights_before = problem.batched_policy_model.get_weights()[1]

        history = problem.train_policy_model(steps_per_epoch=10, epochs=1)
        assert np.isfinite(history.history['loss'][0])
        assert 'policy.01.index' in os.listdir(problem.policy_checkpoint_dir)

        weights_after = problem.batched_policy_model.get_weights()[1]

        assert not np.isclose(weights_before, weights_after).all()

    def test_load_new_policy(self, engine, game):
        problem = game.problem
        assert problem._checkpoint is None

        def get_root_value_pred(problem):
            root = game.get_vertex_for_state(problem.get_initial_state())
            game._evaluate([root])
            value, prior_logits = problem.policy_evaluator(
                problem._get_batched_policy_inputs(root))

            return float(value[0]), prior_logits.numpy()[1:]

        initial_value, initial_priors = get_root_value_pred(problem)

        # This should trigger the checkpoint reloading
        game.run(num_mcts_samples=5)
        assert problem._checkpoint is not None

        next_value, next_priors = get_root_value_pred(problem)

        # Make sure the policy changes the value prediction
        assert not np.isclose(initial_value, next_value)
        assert not np.isclose(initial_priors, next_priors).all()
