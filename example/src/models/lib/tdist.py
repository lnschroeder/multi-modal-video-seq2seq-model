"""
Author of file: Manuel Woellhaf
"""
import tensorflow as tf
from tensorflow import keras


class TimeDistributedReturnSkip(keras.layers.Layer):

    def __init__(self, layer, keyframe=0, name='time_distributed_return_skip', **kwargs):
        super(TimeDistributedReturnSkip, self).__init__(name=name, **kwargs)
        self._layer = layer
        self._keyframe = keyframe
        self._layer.return_skips = True

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        _, skips = self._layer(inputs[:, self._keyframe], training=training) # encode
        input_tensor = tf.reshape(inputs, tf.concat([[-1], input_shape[-3:]], axis=0)) # merge dims
        encodings, _ = self._layer(input_tensor, training=training)
        encodings = tf.reshape(encodings, tf.concat([input_shape[:2], tf.shape(encodings)[1:]], axis=0))
        return encodings, skips


class TimeDistributedRepeatSkip(keras.layers.Layer):

    def __init__(self, layer, name='time_distributed_repeat_skip', **kwargs):
        super(TimeDistributedRepeatSkip, self).__init__(name=name, **kwargs)
        self._layer = layer

    def call(self, inputs, skips, training=None):
        input_shape = tf.shape(inputs)
        input_tensor = tf.reshape(inputs, tf.concat([[-1], input_shape[-1:]], axis=0)) # merge dims
        skips = [tf.repeat(skip, input_shape[1], axis=0) for skip in skips]
        decodings = self._layer([input_tensor, skips], training=training) # decode
        decodings = tf.reshape(decodings, tf.concat([input_shape[:2], tf.shape(decodings)[1:]], axis=0))
        return decodings
