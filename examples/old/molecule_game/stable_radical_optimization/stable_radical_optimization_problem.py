import logging
import os
from typing import (
    Optional,
)

import numpy as np
import psycopg2
import tensorflow as tf
from molecule_game.mol_preprocessor import (
    MolPreprocessor,
    atom_featurizer,
    bond_featurizer,
)
from rdkit.Chem.rdmolfiles import MolFromSmiles
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from examples.old.stable_radical_optimization.run_mcts import predict
from molecule_game.stable_radical_optimization.stable_radical_optimization_state import StableRadicalOptimizationState
from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.molecule.policy.model import build_policy_evaluator
from rlmolecule.molecule.policy.preprocessor import load_preprocessor

logger = logging.getLogger(__name__)

default_preprocessor = MolPreprocessor(atom_features=atom_featurizer,
                                       bond_features=bond_featurizer,
                                       explicit_hs=False)

default_preprocessor.from_json(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../molecule_game/preprocessor.json'))


class StableRadicalOptimizationProblem(AlphaZeroProblem):
    """
    An AlphaZeroProblem that implements a stable radical optimization search.
    """

    def __init__(
            self,
            config: any,
            start_smiles: str,
            preprocessor: Optional[MolPreprocessor] = None,
            preprocessor_data=None,
            policy_checkpoint_dir=None,
    ) -> None:
        # super().__init__(
        #     config.min_reward,
        #     config.pb_c_base,
        #     config.pb_c_init,
        #     config.dirichlet_noise,
        #     config.dirichlet_alpha,
        #     config.dirichlet_x,
        # )
        self._config: any = config
        self._start_smiles: str = start_smiles
        assert (preprocessor is None and preprocessor_data is not None) or \
               (preprocessor is not None and preprocessor_data is None)
        self._preprocessor: MolPreprocessor = preprocessor if preprocessor else load_preprocessor(preprocessor_data)

        policy_evaluator, loaded_checkpoint = build_policy_evaluator(policy_checkpoint_dir)
        self._policy_evaluator: tf.function = policy_evaluator

        if loaded_checkpoint:
            logger.info(f'{self.id}: Loaded checkpoint {loaded_checkpoint}')
        else:
            logger.info(f'{self.id}: No checkpoint found {loaded_checkpoint}')

        # memoizer, start = \
        #     AlphaZeroNode.make_memoized_root_node(
        #         StableRadicalOptimizationState(self, MolFromSmiles(start_smiles), False), self)
        #
        # self._graph_memoizer: NetworkXNodeMemoizer = memoizer
        # self._start: AlphaZeroNode = start
        # self._policy_trainer = build_policy_trainer()
        # self._policy_model = self._policy_trainer.layers[-1].policy_model
        # self._policy_predictions = tf.function(experimental_relax_shapes=True)(self._policy_model.predict_step)
        # self.load_from_checkpoint()  # TODO: does this ever do anything?
        pass

    @property
    def config(self) -> any:
        return self._config

    def get_initial_state(self) -> StableRadicalOptimizationState:
        return StableRadicalOptimizationState(MolFromSmiles(self._start_smiles), self._config, False)

    def get_reward(self, state: StableRadicalOptimizationState, policy_inputs: any) -> float:
        config = self.config
        reward = None

        # TODO: consider factoring out this database memoization code
        with psycopg2.connect(**config.dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select real_reward from {table}_reward where smiles = %s".format(
                        table=config.sql_basename), (state.smiles,))
                result = cur.fetchone()
            if result:
                reward = result[0]

        if reward is None:
            if ((policy_inputs['atom'] == 1).any() |
                    (policy_inputs['bond'] == 1).any()):
                # Node is outside the domain of validity
                # self._true_reward = 0.
                return config.min_reward
            else:
                spins, buried_vol = predict(
                    {key: tf.constant(np.expand_dims(val, 0))
                     for key, val in policy_inputs.items()})

                spins = spins.numpy().flatten()
                buried_vol = buried_vol.numpy().flatten()

                atom_index = int(spins.argmax())
                max_spin = spins[atom_index]
                spin_buried_vol = buried_vol[atom_index]

                atom_type = state.molecule.GetAtomWithIdx(atom_index).GetSymbol()

                # This is a bit of a placeholder; but the range for spin is about 1/50th that
                # of buried volume.
                reward = (1 - max_spin) * 50 + spin_buried_vol

                with psycopg2.connect(**config.dbparams) as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO {table}_reward
                            (smiles, real_reward, atom_type, buried_vol, max_spin, atom_index)
                            values (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING;""".format(table=config.sql_basename), (
                            state.smiles, float(reward), atom_type,  # This should be the real reward
                            float(spin_buried_vol), float(max_spin), atom_index))

            ranked_reward = self._get_ranked_rewards(reward)

            # TODO: do we want to store this here?
            # node._true_reward = reward
            return ranked_reward

    def get_value_and_policy(self, successors: [AlphaZeroVertex]) -> (float, {AlphaZeroVertex: float}):
        values, prior_logits = self._policy_evaluator(self._make_batched_policy_inputs(successors))

        # inputs to policy network
        priors = tf.nn.softmax(prior_logits[1:]).numpy().flatten()

        # Update child nodes with predicted prior_logits
        successor_priors = {node: prior for node, prior in zip(successors, priors)}
        value = float(tf.nn.sigmoid(values[0]))

        return value, successor_priors

    def _make_batched_policy_inputs(self, vertices: [AlphaZeroVertex]) -> {}:
        """
        :return the given verticies policy inputs concatenated together. Used as the inputs for the policy neural
        network.
        """
        policy_inputs = [self._get_policy_inputs(vertex) for vertex in vertices]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}

    def _get_policy_inputs(self, vertex: AlphaZeroVertex) -> {}:
        """
        :return GNN inputs for the node
        """
        # TODO: memoize
        # if self._policy_inputs is None:
        #     self._policy_inputs = self.game.preprocessor.construct_feature_matrices(self)
        # return self._policy_inputs

        # noinspection PyUnresolvedReferences
        return self._preprocessor.construct_feature_matrices(vertex.state.molecule)

    def _get_ranked_rewards(self, reward: float) -> float:
        config = self.config
        with psycopg2.connect(**config.dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute("select count(*) from {table}_game where experiment_id = %s;".format(
                    table=config.sql_basename), (config.experiment_id,))
                n_games = cur.fetchone()[0]

            if n_games < config.reward_buffer_min_size:
                # Here, we don't have enough of a game buffer
                # to decide if the move is good or not
                logging.debug(f"ranked_reward: not enough games ({n_games})")
                return np.random.choice([0., 1.])

            else:
                with conn.cursor() as cur:
                    cur.execute("""
                            select percentile_disc(%s) within group (order by real_reward)
                            from (select real_reward from {table}_game where experiment_id = %s
                                  order by id desc limit %s) as finals
                            """.format(table=config.sql_basename),
                                (config.ranked_reward_alpha, config.experiment_id, config.reward_buffer_max_size))

                    r_alpha = cur.fetchone()[0]

                logging.debug(f"ranked_reward: r_alpha={r_alpha}, reward={reward}")

                if np.isclose(reward, r_alpha):
                    return np.random.choice([0., 1.])

                elif reward > r_alpha:
                    return 1.

                elif reward < r_alpha:
                    return 0.
