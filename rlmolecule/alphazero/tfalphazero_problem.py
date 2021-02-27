from abc import abstractmethod
import itertools
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import sqlalchemy

import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from rlmolecule.alphazero.alphazero_problem import AlphaZeroProblem
from rlmolecule.alphazero.alphazero_vertex import AlphaZeroVertex
from rlmolecule.alphazero.keras_utils import KLWithLogits, TimeCsvLogger
from rlmolecule.tree_search.graph_search_state import GraphSearchState

import tensorflow as tf
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def build_policy_trainer(model: tf.keras.Model, input_masks: dict = {}) -> tf.keras.Model:
    """Returns a wrapper policy model and input masks."""
    value_preds, masked_prior_logits = PolicyWrapper(model, input_masks)(model.inputs)
    return tf.keras.Model(model.inputs, [value_preds, masked_prior_logits])


def get_input_mask_dict(inputs: list, 
                        mask_dict: dict = {},
                        as_tensor=False, 
                        value: float = 0.) -> dict:
    """Returns a dictionary of mask values with type cast to that of the 
    corresponding input layer."""
    _mask_dict = {inp.name: value for inp in inputs}
    _mask_dict.update(mask_dict)
    if as_tensor:
        return {inp.name: tf.constant(_mask_dict[inp.name], dtype=inp.dtype) for inp in inputs}
    else:
        return {inp.name: _mask_dict[inp.name] for inp in inputs}


class PolicyWrapper(layers.Layer):

    def __init__(self,
                 policy_model: tf.keras.Model,
                 mask_dict: dict = {},  # will default to 0 if you don't specify
                 **kwargs):               # **kwargs will go unused currently

        super().__init__(**kwargs)
        self.policy_model = policy_model
        self.mask_dict = get_input_mask_dict(policy_model.inputs, mask_dict, as_tensor=True)

    def build(self, input_shape):
        pass

    def call(self, inputs, mask=None):

        #print("INPUTS", inputs)

        # Get the batch and action dimensions
        shape = tf.shape(inputs[0])
        batch_size = shape[0]
        max_actions_per_node = shape[1]
        flattened_shape = batch_size * max_actions_per_node
        print(shape, batch_size, max_actions_per_node, flattened_shape)
        
        # Flatten the inputs for running individually through the policy model
        flattened_inputs = []
        for inp in inputs:
            new_shape = [flattened_shape, -1]
            if inp.shape.ndims > 2:
                new_shape += tf.shape(inp)[2:]
                #print("SHAPE, NEW SHAPE", tf.shape(inp), new_shape)
            flattened_inputs.append(tf.reshape(inp, new_shape))
            
        # Get the flat value and prior_logit predictions
        #print("FLATTENED_INPUTS", flattened_inputs)
        flat_values_logits, flat_prior_logits = self.policy_model(flattened_inputs)

        # We put the parent node first in our batch inputs, so this slices
        # the value prediction for the parent
        value_preds = tf.reshape(flat_values_logits, [batch_size, max_actions_per_node, -1])[:, 0, 0]
        
        # Next we get a mask to see where we have valid actions and replace priors for
        # invalid actions with negative infinity (these get zeroed out after softmax).
        # We also only return prior_logits for the child nodes (not the first entry).
        prior_logits = tf.reshape(flat_prior_logits, [batch_size, max_actions_per_node])
        action_mask = tf.cast(tf.ones_like(prior_logits), tf.bool)
        for i, inp in enumerate(inputs):
            inp = tf.reshape(inp, [batch_size, max_actions_per_node, -1])
            new_mask = tf.reduce_all(
                tf.not_equal(inp, 
                             self.mask_dict[self.policy_model.inputs[i].name]),
                             axis=-1)
            action_mask = tf.logical_and(action_mask, new_mask)

        # Apply the mask
        masked_prior_logits = tf.where(
            action_mask, 
            prior_logits, 
            tf.ones_like(prior_logits) * prior_logits.dtype.min)[:, 1:]

        return value_preds, masked_prior_logits


class TFAlphaZeroProblem(AlphaZeroProblem):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 model: tf.keras.Model,
                 mask_dict: dict = {},
                 policy_checkpoint_dir: Optional[str] = None,
                 **kwargs
                 ) -> None:

        super(TFAlphaZeroProblem, self).__init__(engine, **kwargs)
        self.mask_dict = get_input_mask_dict(model.inputs, mask_dict, as_tensor=False)
        self.policy_model = build_policy_trainer(model, self.mask_dict)
        self.policy_checkpoint_dir = policy_checkpoint_dir
        policy_model_layer = self.policy_model.layers[-1].policy_model
        self.policy_evaluator = tf.function(experimental_relax_shapes=True)(policy_model_layer.predict_step)
        self._checkpoint = None

    def initialize_run(self):
        """
        Load the most recent policy checkpoint
        """
        super().initialize_run()

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

    @abstractmethod
    def get_policy_inputs(self, state: GraphSearchState) -> Dict:
        pass

    def _get_batched_policy_inputs(self, parent: AlphaZeroVertex) -> Dict:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """
        # Get the list of policy inputs
        policy_inputs = [self.get_policy_inputs(vertex.state)
                         for vertex in itertools.chain((parent,), parent.children)]

        print("POLICY INPUTS", policy_inputs)

        # Return the padded values, using the input_mask dict from the policy wrapper.
        return {key: pad_sequences([elem[key] for elem in policy_inputs], 
                                    padding='post',
                                    value=self.mask_dict[key])
                for key in policy_inputs[0].keys()}

    def get_value_and_policy(self, parent: AlphaZeroVertex) -> Tuple[float, dict]:

        print("PARENT", parent)
        values, prior_logits = self.policy_evaluator(self._get_batched_policy_inputs(parent))

        # Softmax the child priors.  Be careful here that you're slicing all needed
        # dimensions, otherwise you can end up with elementwise softmax (i.e., all 1's).
        # TODO:  Did Dave break this for peter?  I'm getting only a 2d output, not sure
        # if this will always be the case.  Might need another level inspection
        # here for output shape, or enforce an output shape (?).
        priors = tf.nn.softmax(prior_logits[1:, 0]).numpy().flatten()

        # Update child nodes with predicted prior_logits
        children_priors = {vertex: prior for vertex, prior in zip(parent.children, priors)}
        value = float(tf.nn.sigmoid(values[0]))

        return value, children_priors


    def _get_policy_inputs_from_serialized_parent(
            self,
            serialized_parent: tf.Tensor) -> Tuple[dict, dict]:
        
        parent = self.get_initial_state().deserialize(serialized_parent.numpy().decode())

        policy_inputs = [self.get_policy_inputs(parent)
                         for state in itertools.chain((parent,), parent.get_next_actions())]

        policy_inputs = {key: pad_sequences(
                                [elem[key] for elem in policy_inputs], 
                                padding='post',
                                value=self.mask_dict[key])
                         for key in policy_inputs[0]}

        return policy_inputs['position'], policy_inputs['steps']


    def _create_dataset(self) -> tf.data.Dataset:
        """
        Creates a tensorflow dataset pipeline to batch game positions from the replay buffer into

        :param problem:
        :return:
        """

        def get_policy_inputs_tf(parent, reward, visit_probabilities,
                                 problem: TFAlphaZeroProblem) -> dict:
            inputs = tf.py_function(
                problem._get_policy_inputs_from_serialized_parent, 
                inp=[parent],
                Tout=[inp.dtype for inp in self.policy_model.inputs])
            print([tf.shape(t) for t in inputs])
            inputs = [tf.expand_dims(t, 0) for t in inputs]  # is this the same as adding None axis?
            result = {inp.name: value for inp, value in zip(self.policy_model.inputs, inputs)}
            print("RESULT", result)
            return result, (reward, visit_probabilities)

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
