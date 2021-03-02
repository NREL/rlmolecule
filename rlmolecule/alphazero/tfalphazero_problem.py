import itertools
import logging
import os
import time
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import sqlalchemy
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.alphazero.tf_keras_policy import (KLWithLogits, PolicyWrapper, TimeCsvLogger, align_input_names)
from rlmolecule.tree_search.graph_search_state import GraphSearchState

logger = logging.getLogger(__name__)


class TFAlphaZeroProblem(AlphaZeroProblem):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 policy_checkpoint_dir: str = 'policy_checkpoints',
                 **kwargs
                 ) -> None:

        super(TFAlphaZeroProblem, self).__init__(engine, **kwargs)
        self.mask_dict = self.get_policy_mask()
        self.batched_policy_model = PolicyWrapper.build_policy_model(self.policy_model(), self.mask_dict)
        self.input_names = align_input_names(self.batched_policy_model.inputs, self.mask_dict)
        self.policy_checkpoint_dir = policy_checkpoint_dir
        single_position_policy = self.batched_policy_model.layers[-1].single_position_policy
        self.policy_evaluator = tf.function(experimental_relax_shapes=True)(single_position_policy.predict_step)
        self._checkpoint = None

    @abstractmethod
    def policy_model(self) -> 'tf.keras.Model':
        pass

    @abstractmethod
    def get_policy_inputs(self, state: GraphSearchState) -> {str: np.ndarray}:
        pass

    def get_policy_mask(self) -> {str: Optional[float]}:
        initial_inputs = self.get_policy_inputs(self.get_initial_state())
        return {key: np.array(0, dtype=val.dtype) for key, val in initial_inputs.items()}

    def initialize_run(self):
        """
        Load the most recent policy checkpoint
        """
        super().initialize_run()

        new_checkpoint = tf.train.latest_checkpoint(self.policy_checkpoint_dir)
        if new_checkpoint != self._checkpoint:
            self._checkpoint = new_checkpoint
            status = self.batched_policy_model.load_weights(self._checkpoint)
            status.assert_existing_objects_matched()
            logger.info(f'Loaded checkpoint {self._checkpoint}')
        elif new_checkpoint == self._checkpoint:
            logger.info(f'Skipping already loaded {self._checkpoint}')
        else:
            logger.info('No checkpoint found')

    def _get_batched_policy_inputs(self, parent: AlphaZeroVertex) -> Dict:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """
        # Get the list of policy inputs
        policy_inputs = [self.get_policy_inputs(vertex.state)
                         for vertex in itertools.chain((parent,), parent.children)]

        # print("POLICY INPUTS", policy_inputs)

        # Return the padded values, using the input_mask dict from the policy wrapper.
        return {key: pad_sequences([elem[key] for elem in policy_inputs],
                                   padding='post',
                                   value=self.mask_dict[key])
                for key in policy_inputs[0].keys()}

    def get_value_and_policy(self, parent: AlphaZeroVertex) -> Tuple[float, dict]:

        # print("PARENT", parent)
        values, prior_logits = self.policy_evaluator(self._get_batched_policy_inputs(parent))

        # Softmax the child priors.  Be careful here that you're slicing all needed
        # dimensions, otherwise you can end up with elementwise softmax (i.e., all 1's).
        # if this will always be the case.  Might need another level inspection
        # here for output shape, or enforce an output shape (?).
        priors = tf.nn.softmax(prior_logits[1:, 0]).numpy().flatten()

        # Update child nodes with predicted prior_logits
        children_priors = {vertex: prior for vertex, prior in zip(parent.children, priors)}
        value = float(tf.nn.sigmoid(values[0]))

        return value, children_priors

    def _get_policy_inputs_from_serialized_parent(
            self,
            serialized_parent: tf.Tensor) -> Tuple:

        parent = self.get_initial_state().deserialize(serialized_parent.numpy().decode())

        # todo: any clever way to combine this with _get_batched_policy_inputs?
        policy_inputs = [
            self.get_policy_inputs(state) for state in itertools.chain((parent,), parent.get_next_actions())]

        policy_inputs = {key: pad_sequences(
            [elem[key] for elem in policy_inputs],
            padding='post',
            value=self.mask_dict[key])
            for key in policy_inputs[0]}

        sorted_policy_inputs = tuple([policy_inputs[input_name] for input_name in self.input_names])

        return sorted_policy_inputs

    def _create_dataset(self) -> tf.data.Dataset:
        """
        Creates a tensorflow dataset pipeline to batch game positions from the replay buffer into
        """

        def get_policy_inputs_tf(parent, reward, visit_probabilities,
                                 problem: TFAlphaZeroProblem):
            input_layers = self.batched_policy_model.inputs

            inputs = tf.py_function(
                problem._get_policy_inputs_from_serialized_parent,
                inp=[parent],
                Tout=[inp.dtype for inp in input_layers])
            # print([tf.shape(t) for t in inputs])
            for inp, input_layer in zip(inputs, input_layers):
                inp.set_shape(input_layer.shape[1:])

            result = {name: value for name, value in zip(self.input_names, inputs)}
            # print("RESULT", result)
            return result, (reward, visit_probabilities)

        dataset = tf.data.Dataset.from_generator(
            self.iter_recent_games,
            output_shapes=((), (), (None,)),
            output_types=(tf.string, tf.float32, tf.float32)) \
            .repeat() \
            .shuffle(self.max_buffer_size) \
            .map(partial(get_policy_inputs_tf, problem=self),
                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .padded_batch(self.batch_size,
                          padding_values=(
                              self.mask_dict,
                              (0., 0.))) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        # Need to confirm that we can keep the padding values for
        # (reward, visit_probabilities) as (0., 0.).

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
        self.batched_policy_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True), KLWithLogits()])

        logger.info("Policy trainer: starting training")

        return self.batched_policy_model.fit(
            dataset, steps_per_epoch=steps_per_epoch,
            epochs=epochs, callbacks=[model_checkpoint, csv_logger], **kwargs)
