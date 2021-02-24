from abc import abstractmethod
import itertools
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from numpy import array

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


def make_input_mask(values: tf.Tensor, mask_value: float) -> tf.Tensor:
    """Masks a given tensor based on values along all but the batch and action axes."""
    shape = tf.shape(values)
    values = tf.reshape(values, (shape[0], shape[1], -1))
    return tf.reduce_all(tf.not_equal(values, mask_value), axis=-1)

def make_action_mask(inputs: List[tf.Tensor], mask_values: list) -> tf.Tensor:
    """Returns an action mask for a given set of input tensors."""
    batch_size, max_actions_per_node = tf.shape(inputs[0])[:2]
    action_mask = tf.constant(True, shape=[batch_size, max_actions_per_node], dtype=bool)
    for i, inp in enumerate(inputs):
        new_mask =  make_input_mask(inp, mask_values[i])
        action_mask = tf.logical_and(action_mask, new_mask)
    return action_mask

def flatten_batch_and_action_axes(x: tf.Tensor) -> tf.Tensor:
    """Returns a tensor flattened along the first two axes."""
    shape = tf.shape(x)
    return tf.reshape(x, [shape[0] * shape[1], *shape[2:]])
    
def build_policy_trainer(model: tf.keras.Model) -> tf.keras.Model:
    """Returns a wrapper policy model."""
    value_preds, masked_prior_logits = PolicyWrapper(model)(model.inputs)
    policy_trainer = tf.keras.Model(model.inputs, [value_preds, masked_prior_logits])
    return policy_trainer


class PolicyWrapper(layers.Layer):

    def __init__(self,
                 policy_model: tf.keras.Model,
                 input_masks: dict = {},  # will default to 0 if you don't specify
                 **kwargs):

        super().__init__(**kwargs)
        self._policy_model = policy_model

        # Here we create default 0-values for masks, and then update with any 
        # user supplied masks keyed on the input layer names.
        self._input_masks = {inp.name: 0. for inp in self._policy_model.inputs}
        self._input_masks.update(input_masks)


    def build(self, input_shape):
        pass


    def call(self, inputs, mask):

        # Get the batch and action dimensions
        shape = tf.shape(inputs[0])
        batch_size = shape[0]
        max_actions_per_node = shape[1]

        # Flatten the inputs for running individually through the policy model
        flattened_inputs = [flatten_batch_and_action_axes(inp) for inp in inputs]

        # Get the flat value and prior_logit predictions
        flat_values_logits, flat_prior_logits = self.policy_model(flattened_inputs)

        # We put the parent node first in our batch inputs, so this slices
        # the value prediction for the parent
        value_preds = tf.reshape(flat_values_logits, [batch_size, max_actions_per_node, -1])[:, 0, 0]

        # Next we get a mask to see where we have valid actions and replace priors for
        # invalid actions with negative infinity (these get zeroed out after softmax).
        # We also only return prior_logits for the child nodes (not the first entry)
        mask_values = [self._input_masks[inp.name] for inp in self._policy_model.inputs]
        action_mask = make_action_mask(inputs, mask_values)
        prior_logits = tf.reshape(flat_prior_logits, [batch_size, max_actions_per_node])
        masked_prior_logits = tf.where(
            action_mask, 
            prior_logits,
            tf.ones_like(prior_logits) * prior_logits.dtype.min)[:, 1:]

        return value_preds, masked_prior_logits

    @property
    def input_masks(self):
        return self._input_masks


class TFAlphaZeroProblem(AlphaZeroProblem):
    def __init__(self,
                 engine: sqlalchemy.engine.Engine,
                 model: tf.keras.Model,
                 model_input_masks: dict = {},
                 policy_checkpoint_dir: Optional[str] = None,
                 **kwargs
                 ) -> None:

        super(TFAlphaZeroProblem, self).__init__(engine, **kwargs)
        #self.policy_model = build_policy_trainer(hallway_size, hidden_layers, hidden_dim)
        self.model_input_masks = model_input_masks
        self.policy_model = build_policy_trainer(model)
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
    def _get_network_inputs(self, state: GraphSearchState) -> Dict:
        pass

    def _get_batched_network_inputs(self, parent: AlphaZeroVertex) -> Dict:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """
        # Get the list of policy inputs
        policy_inputs = [self._get_network_inputs(vertex.state)
                         for vertex in itertools.chain((parent,), parent.children)]

        # Return the padded values, using the input_mask dict from the policy wrapper.
        return {key: pad_sequences([elem[key] for elem in policy_inputs], 
                                    padding='post',
                                    value=self.policy_model.input_mask[key])
                for key in policy_inputs[0].keys()}

    def get_value_and_policy(self, parent: AlphaZeroVertex) -> Tuple[float, dict]:

        values, prior_logits = self.policy_evaluator(self._get_batched_network_inputs(parent))

        # Softmax the child priors.  Be careful here that you're slicing all needed
        # dimensions, otherwise you can end up with elementwise softmax (i.e., all 1's).
        priors = tf.nn.softmax(prior_logits[1:, 0, 0]).numpy().flatten()

        # Update child nodes with predicted prior_logits
        children_priors = {vertex: prior for vertex, prior in zip(parent.children, priors)}
        value = float(tf.nn.sigmoid(values[0]))

        return value, children_priors

    ## TODO:  Dave, pick up here

    def _get_network_inputs_from_serialized_parent(
            self,
            serialized_parent: tf.Tensor) -> Tuple[dict, dict]:
        
        # How to do this without having to hardcode in the state class?
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
                problem._get_network_inputs_from_serialized_parent, 
                inp=[parent],
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
