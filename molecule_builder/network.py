import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, GlobalAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras import backend as K

import os
import glob

from config import AlphaZeroConfig

CONFIG = AlphaZeroConfig()


class Network:


    def __init__(self, checkpoint_dir=None):
        self.checkpoint_dir = checkpoint_dir
        self.create_model()


    def create_model(self):
        """Creates the Keras model."""

        # Pull out for convenience
        fingerprint_dim = CONFIG.fingerprint_dim
        max_actions = CONFIG.max_next_mols
        num_hidden_units = CONFIG.num_hidden_units

        # Kernel regularizer and initializer for generic dense layers
        kreg = tf.keras.regularizers.l2(l=CONFIG.l2_regularization_coef)
        kini = tf.keras.initializers.Zeros()
        def dense():
            return Dense(num_hidden_units, kernel_regularizer=kreg, kernel_initializer=kini)

        # Inputs
        mol = Input(shape=[None, fingerprint_dim], name="mol")
        next_mols = Input(shape=[None, max_actions, fingerprint_dim], name="next_mols")
        action_mask = Input(shape=[None, max_actions], name="action_mask")

        # Value network
        x = dense()(mol)
        x = dense()(x)
        x = GlobalAveragePooling1D()(x)
        v = Dense(1, activation="tanh", name="v", kernel_regularizer=kreg, 
                    kernel_initializer=kini)(x)

        # Policy network
        y = dense()(mol)
        y = dense()(y)
        action_embed = Dense(fingerprint_dim, kernel_regularizer=kreg, 
            kernel_initializer=kini, activation="linear", name="action_embed")(y)
        intent_vector = tf.expand_dims(action_embed, 2)
        pi_logits = tf.reduce_sum(next_mols * intent_vector, axis=-1)
        inf_mask = tf.maximum(K.log(action_mask), tf.float32.min)
        pi_logits = pi_logits + inf_mask
        pi_logits = GlobalAveragePooling1D()(pi_logits)
        pi_logits = Lambda(lambda x: x, name="pi_logits")(pi_logits)

        self.model = Model(inputs=[mol, next_mols, action_mask], outputs=[v, pi_logits])


    def compile(self):
        """Compile the model.  This is needed for training but not for inference."""
        self.model.compile(
            optimizer="adam", 
            loss={"v": tf.keras.losses.MSE, 
                  "pi_logits": tf.nn.softmax_cross_entropy_with_logits})


    def load_model(self, model_dir):
        """Update the latest model weights."""
        if self.checkpoint_dir is None:
            pass
        else:
            # Actually do something here! 
            glob_pattern = os.path.join(model_dir, 'cp*')
            file_list = glob.glob(glob_pattern, recursive=False)
            if file_list:
                latest = tf.train.latest_checkpoint(model_dir)
                self.model.load_weights(latest)
                #new_model = tf.keras.models.load_model(latest_file, compile=False)


    def inference(self, mol, next_mols, action_mask):
        """Forward pass."""
        v, pi = self.model([mol[None, None, :], next_mols[None, None, :], action_mask[None, None, :]])
        return tf.squeeze(v), tf.squeeze(pi)

if __name__ == "__main__":
    current_path = os.getcwd()
    checkpoint_dir = os.path.join(current_path, 'saved_models')
    network = Network(checkpoint_dir)
    network.load_model(checkpoint_dir)
    print(network.model.summary())
