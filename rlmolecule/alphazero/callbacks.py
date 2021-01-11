from datetime import datetime

import tensorflow as tf


class TimeCsvLogger(tf.keras.callbacks.CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        super(TimeCsvLogger, self).on_epoch_end(epoch, logs)
