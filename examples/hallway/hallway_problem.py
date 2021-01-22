import itertools
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional

from numpy import array

import sqlalchemy
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.alphazero.keras_utils import KLWithLogits, TimeCsvLogger

from hallway_state import HallwayState
from model import build_policy_trainer

logger = logging.getLogger(__name__)


class HallwayAlphaZeroProblem(AlphaZeroProblem):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 policy_checkpoint_dir: Optional[str] = None,
                 hallway_size: int = 5,
                 max_steps: int = 32,
                 hidden_layers: int = 3,
                 hidden_dim: int = 16,
                 **kwargs
                 ) -> None:

        super(HallwayAlphaZeroProblem, self).__init__(engine, **kwargs)
        hallway_size = hallway_size
        max_steps = max_steps
        self.policy_model = build_policy_trainer(hallway_size, hidden_layers, hidden_dim)
        self.policy_checkpoint_dir = policy_checkpoint_dir
        policy_model_layer = self.policy_model.layers[-1].policy_model
        self.policy_evaluator = tf.function(experimental_relax_shapes=True)(policy_model_layer.predict_step)
        self._checkpoint = None


    def initialize_run(self):
        """
        Load the most recent policy checkpoint
        """
        super(HallwayAlphaZeroProblem, self).initialize_run()

        if self.policy_checkpoint_dir:
            new_checkpoint = tf.train.latest_checkpoint(self.policy_checkpoint_dir)
            if new_checkpoint != self._checkpoint:
                self._checkpoint = new_checkpoint
                status = self.policy_model.load_weights(self._checkpoint)
                status.assert_existing_objects_matched()
                logger.info(f'Loaded checkpoint {self._checkpoint}')
            elif new_checkpoint == self._checkpoint:
                logger.info(f'Skipping already loaded {self._checkpoint}')
            else:
                logger.info('No checkpoint found')

    def _get_network_inputs(self, state: HallwayState) -> Dict:
        return {"position": array([state.position]), "steps": array([state.steps])}

    def _get_batched_network_inputs(self, parent: AlphaZeroVertex) -> Dict:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """
        policy_inputs = [self._get_network_inputs(vertex.state)
                         for vertex in itertools.chain((parent,), parent.children)]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}

    def get_value_and_policy(self, parent: AlphaZeroVertex) -> (float, {AlphaZeroVertex: float}):

        values, prior_logits = self.policy_evaluator(self._get_batched_network_inputs(parent))

        # Softmax the child priors.  Be careful here that you're slicing all needed
        # dimensions, otherwise you can end up with elementwise softmax (i.e., all 1's).
        priors = tf.nn.softmax(prior_logits[1:, 0, 0]).numpy().flatten()

        # Update child nodes with predicted prior_logits
        children_priors = {vertex: prior for vertex, prior in zip(parent.children, priors)}
        value = float(tf.nn.sigmoid(values[0]))

        return value, children_priors


    def _get_network_inputs_from_serialized_parent(
            self,
            serialized_parent: tf.Tensor) -> ({}, {}):

        parent = HallwayState.deserialize(serialized_parent.numpy().decode())

        policy_inputs = [self._get_network_inputs(parent)
                         for state in itertools.chain((parent,), parent.get_next_actions())]

        policy_inputs = {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                         for key in policy_inputs[0].keys()}

        return policy_inputs['position'], policy_inputs['steps']


    def _create_dataset(self) -> tf.data.Dataset:
        """
        Creates a tensorflow dataset pipeline to batch game positions from the replay buffer into

        :param problem:
        :return:
        """

        def get_policy_inputs_tf(parent, reward, visit_probabilities,
                                 problem: HallwayAlphaZeroProblem) -> {}:
            position, steps = tf.py_function(
                problem._get_network_inputs_from_serialized_parent, inp=[parent],
                Tout=[tf.int64, tf.int64])
            position.set_shape([None, 1])
            steps.set_shape([None, 1])
            return {"position": position, "steps": steps}, (reward, visit_probabilities)

        dataset = tf.data.Dataset.from_generator(
            self.iter_recent_games,
            output_shapes=((), (), (None, )),
            output_types=(tf.string, tf.float32, tf.float32)) \
            .repeat() \
            .shuffle(self.max_buffer_size) \
            .map(partial(get_policy_inputs_tf, problem=self),
                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .padded_batch(self.batch_size,
                          padding_values=(
                              {"position": tf.constant(0, dtype=tf.int64), # Not sure what this should be
                               "steps": tf.constant(0, dtype=tf.int64)},
                              (0., 0.))) \
            .prefetch(tf.data.experimental.AUTOTUNE)
            
        return dataset

    def train_policy_model(
            self,
            steps_per_epoch: int = 750,
            lr: float = 1E-3,
            epochs: int = int(1E4),
            game_count_delay: int = 30,
            **kwargs) -> tf.keras.callbacks.History:

        # wait to start training until enough games have occurred
        while len(list(self.iter_recent_games())) < self.min_buffer_size:
            logging.info(f"Policy trainer: waiting, not enough games found ({len(list(self.iter_recent_games()))})")
            time.sleep(game_count_delay)

        # Create the games dataset
        dataset = self._create_dataset()
        print(dataset)

        # Create a callback to store optimized models at given frequencies
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.policy_checkpoint_dir, 'policy.{epoch:02d}'),
            save_best_only=False, save_weights_only=True)

        # Log the time as well as the epoch to synchronize with the game rewards
        csv_logger = TimeCsvLogger(
            os.path.join(self.policy_checkpoint_dir, 'log.csv'),
            separator=',', append=False)

        # Ensure the the policy checkpoint directory exists
        Path(self.policy_checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Compile the model with a loss function and optimizer
        self.policy_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True), KLWithLogits()])

        logger.info("Policy trainer: starting training")

        return self.policy_model.fit(
            dataset, steps_per_epoch=steps_per_epoch,
            epochs=epochs, callbacks=[model_checkpoint, csv_logger], **kwargs)
