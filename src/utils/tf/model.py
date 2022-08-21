"""
Author of file: Manuel Woellhaf
"""
import os

import tensorflow as tf

from keras.engine import data_adapter


class Model(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        self._log_dir = kwargs.pop('log_dir', None)
        super(Model, self).__init__(*args, **kwargs)
        if self._log_dir is not None:
            self._gradient_writer = tf.summary.create_file_writer(os.path.join(self._log_dir, "tb", "gradients"))

    def train_step(self, data):
        """
        This train step is mostly exactly the same code as in the original implementation,
        adding only logging of the gradient histogram.
        """

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        if self.loss and y is None:
            raise TypeError(
                f'Target data is missing. Your model has `loss`: {self.loss}, '
                'and therefore expects target data to be passed in `fit()`.'
            )

        gradients = tape.gradient(loss, self.trainable_variables)

        if self._gradient_writer:
            with self._gradient_writer.as_default():
                for weights, grads in zip(self.trainable_variables, gradients):
                    tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads, step=self._train_counter)
            self._gradient_writer.flush()

        # Run backwards pass.
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return return_metrics
