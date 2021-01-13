from datetime import datetime

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils


class TimeCsvLogger(tf.keras.callbacks.CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        super(TimeCsvLogger, self).on_epoch_end(epoch, logs)


def kl_with_logits(y_true, y_pred) -> tf.Tensor:
    """ It's typically more numerically stable *not* to perform the softmax,
    but instead define the loss based on the raw logit predictions. This loss
    function corrects a tensorflow omission where there isn't a KLD loss that
    accepts raw logits. """

    # Mask nan values in y_true with zeros
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))

    return (
            tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True) -
            tf.keras.losses.categorical_crossentropy(y_true, y_true, from_logits=False))


class KLWithLogits(LossFunctionWrapper):
    """ Keras sometimes wants these loss function wrappers to define how to
    reduce the loss over variable batch sizes """

    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='kl_with_logits'):
        super(KLWithLogits, self).__init__(
            kl_with_logits,
            name=name,
            reduction=reduction)