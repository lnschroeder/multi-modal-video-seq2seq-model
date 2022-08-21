"""
Author of file: Manuel Woellhaf
"""
import tensorflow as tf
from tensorflow import keras


class dcgan_block(tf.keras.layers.Layer):

    def __init__(self, filters, ksize=4, strides=2, activation='leaky_relu', padding='same', use_bias=False, name='dcgan_block', **kwargs):
        super(dcgan_block, self).__init__(name=name, **kwargs)
        self.conv = keras.layers.Conv2D(filters, ksize, strides, padding=padding, use_bias=use_bias)
        self.bn = keras.layers.BatchNormalization()
        if activation == 'leaky_relu':
            self.activation = keras.layers.LeakyReLU(alpha=0.3)
        else:
            self.activation = keras.layers.Activation(activation)

    def call(self, input_tensor, training=None):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class dcgan_block_up(tf.keras.layers.Layer):

    def __init__(self, filters, ksize=4, strides=2, activation='leaky_relu', padding='same', use_bias=False, name='dcgan_block_up', **kwargs):
        super(dcgan_block_up, self).__init__(name=name, **kwargs)
        self.conv = keras.layers.Conv2DTranspose(filters, ksize, strides, padding=padding, use_bias=use_bias)
        self.bn = keras.layers.BatchNormalization()
        if activation == 'leaky_relu':
            self.activation = keras.layers.LeakyReLU(alpha=0.3)
        else:
            self.activation = keras.layers.Activation(activation)

    def call(self, input_tensor, training=None):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class dcgan_encoder_128(tf.keras.Model):
    """
    ~ 7 Mio trainable parameters.
    """
    def __init__(self, dim_out, filters=64, activation='linear', return_skips=False, name='dcgan_encoder', **kwargs):
        super(dcgan_encoder_128, self).__init__(name=name, **kwargs)
        self.dim_out = dim_out

        self.b1 = dcgan_block(filters)
        self.b2 = dcgan_block(filters * 2)
        self.b3 = dcgan_block(filters * 4)
        self.b4 = dcgan_block(filters * 8)
        self.b5 = dcgan_block(filters * 8)
        self.b6 = dcgan_block(dim_out, strides=1, activation=activation, padding='valid')

        self.return_skips = return_skips

    def call(self, input_tensor, training=None, **kwargs):

        h1 = self.b1(input_tensor, training=training)
        h2 = self.b2(h1, training=training)
        h3 = self.b3(h2, training=training)
        h4 = self.b4(h3, training=training)
        h5 = self.b5(h4, training=training)
        h6 = self.b6(h5, training=training)

        if self.return_skips:
            return tf.squeeze(h6), [h1, h2, h3, h4, h5]
        else:
            return tf.squeeze(h6)

    def compute_output_shape(self, input_shape):
        if self.return_skips:
            return [
                tf.TensorShape([input_shape[0], self.dim_out]),
                [
                    tf.TensorShape([input_shape[0], 64, 64,  64]),
                    tf.TensorShape([input_shape[0], 32, 32, 128]),
                    tf.TensorShape([input_shape[0], 16, 16, 256]),
                    tf.TensorShape([input_shape[0],  8,  8, 512]),
                    tf.TensorShape([input_shape[0],  4,  4, 512])
                ]
            ]
        else:
            return tf.TensorShape([input_shape[0], self.dim_out])


class dcgan_decoder_128(tf.keras.Model):
    """
    ~ 7 Mio trainable parameters for skip='residual' and -- Mio for skip=True.
    """
    def __init__(self, channels=3, filters=64, activation='linear', skip=False, name='dcgan_decoder', **kwargs):
        super(dcgan_decoder_128, self).__init__(name=name, **kwargs)
        self.channels = channels

        self.b1 = dcgan_block_up(filters * 8, strides=1, padding='valid')
        self.b2 = dcgan_block_up(filters * 8)
        self.b3 = dcgan_block_up(filters * 4)
        self.b4 = dcgan_block_up(filters * 2)
        self.b5 = dcgan_block_up(filters)
        self.b6 = keras.layers.Conv2DTranspose(channels, 4, 2, padding='same', activation=activation)

        if not skip:
            self.skip_connector = keras.layers.Lambda(lambda x: x[0])
        elif skip == 'residual':
            self.skip_connector = keras.layers.Add()
        else:
            self.skip_connector = keras.layers.Concatenate(axis=-1)

        self.skip = skip

    def call(self, input_tensor, training=None, **kwargs):

        if not self.skip:
            x, skip = input_tensor, [None]*5
        else:
            x, skip = input_tensor

        x = x[:, tf.newaxis, tf.newaxis, :]
        x = self.b1(x, training=training)
        x = self.b2(self.skip_connector([x, skip[4]]), training=training)
        x = self.b3(self.skip_connector([x, skip[3]]), training=training)
        x = self.b4(self.skip_connector([x, skip[2]]), training=training)
        x = self.b5(self.skip_connector([x, skip[1]]), training=training)
        x = self.b6(self.skip_connector([x, skip[0]]), training=training)

        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], 128, 128, self.channels])



class dcgan_encoder_64(tf.keras.Model):

    def __init__(self, dim_out, filters=64, activation='linear', return_skips=False, name='dcgan_encoder', **kwargs):
        super(dcgan_encoder_64, self).__init__(name=name, **kwargs)
        self.dim_out = dim_out

        self.b1 = dcgan_block(filters)
        self.b2 = dcgan_block(filters * 2)
        self.b3 = dcgan_block(filters * 4)
        self.b4 = dcgan_block(filters * 8)
        self.b5 = dcgan_block(dim_out, strides=1, activation=activation, padding='valid')

        self.return_skips = return_skips

    def call(self, input_tensor, training=None, **kwargs):
        h1 = self.b1(input_tensor, training=training)
        h2 = self.b2(h1, training=training)
        h3 = self.b3(h2, training=training)
        h4 = self.b4(h3, training=training)
        h5 = self.b5(h4, training=training)

        if self.return_skips:
            return h5[:, 0, 0, :], [h1, h2, h3, h4]
        else:
            return h5[:, 0, 0, :]

    def compute_output_shape(self, input_shape):
        if self.return_skips:
            return [
                tf.TensorShape([input_shape[0], self.dim_out]),
                [
                    tf.TensorShape([input_shape[0], 32, 32,  64]),
                    tf.TensorShape([input_shape[0], 16, 16, 128]),
                    tf.TensorShape([input_shape[0],  8,  8, 256]),
                    tf.TensorShape([input_shape[0],  4,  4, 512])
                ]
            ]
        else:
            return tf.TensorShape([input_shape[0], self.dim_out])


class dcgan_decoder_64(tf.keras.Model):

    def __init__(self, channels=3, filters=64, activation='linear', skip=False, name='dcgan_decoder', **kwargs):
        super(dcgan_decoder_64, self).__init__(name=name, **kwargs)
        self.channels = channels

        self.b1 = dcgan_block_up(filters * 8, strides=1, padding='valid')
        self.b2 = dcgan_block_up(filters * 4)
        self.b3 = dcgan_block_up(filters * 2)
        self.b4 = dcgan_block_up(filters)
        self.b5 = keras.layers.Conv2DTranspose(channels, 4, 2, padding='same', activation=activation)

        if not skip:
            self.skip_connector = keras.layers.Lambda(lambda x: x[0])
        elif skip == 'residual':
            self.skip_connector = keras.layers.Add()
        else:
            self.skip_connector = keras.layers.Concatenate(axis=-1)

        self.skip = skip

    def call(self, input_tensor, training=None, **kwargs):

        if not self.skip:
            x, skip = input_tensor, [None]*4
        else:
            x, skip = input_tensor

        x = x[:, tf.newaxis, tf.newaxis, :]
        x = self.b1(x, training=training)
        x = self.b2(self.skip_connector([x, skip[3]]), training=training)
        x = self.b3(self.skip_connector([x, skip[2]]), training=training)
        x = self.b4(self.skip_connector([x, skip[1]]), training=training)
        x = self.b5(self.skip_connector([x, skip[0]]), training=training)

        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], 64, 64, self.channels])


class dcgan_encoder_32(tf.keras.Model):

    def __init__(self, dim_out, filters=64, activation='linear', return_skips=False, name='dcgan_encoder', **kwargs):
        super(dcgan_encoder_32, self).__init__(name=name, **kwargs)
        self.dim_out = dim_out

        self.b1 = dcgan_block(filters)
        self.b2 = dcgan_block(filters * 2)
        self.b3 = dcgan_block(filters * 4)
        self.b4 = dcgan_block(dim_out, strides=1, activation=activation, padding='valid')

        self.return_skips = return_skips

    def call(self, input_tensor, training=None, **kwargs):
        h1 = self.b1(input_tensor, training=training)
        h2 = self.b2(h1, training=training)
        h3 = self.b3(h2, training=training)
        h4 = self.b4(h3, training=training)

        if self.return_skips:
            return h4[:, 0, 0, :], [h1, h2, h3]
        else:
            return h4[:, 0, 0, :]

    def compute_output_shape(self, input_shape):
        if self.return_skips:
            return [
                tf.TensorShape([input_shape[0], self.dim_out]),
                [
                    tf.TensorShape([input_shape[0], 16, 16,  64]),
                    tf.TensorShape([input_shape[0],  8,  8, 128]),
                    tf.TensorShape([input_shape[0],  4,  4, 256])
                ]
            ]
        else:
            return tf.TensorShape([input_shape[0], self.dim_out])


class dcgan_decoder_32(tf.keras.Model):

    def __init__(self, channels=3, filters=64, activation='linear', skip=False, name='dcgan_decoder', **kwargs):
        super(dcgan_decoder_32, self).__init__(name=name, **kwargs)
        self.channels = channels

        self.b1 = dcgan_block_up(filters * 4, strides=1, padding='valid')
        self.b2 = dcgan_block_up(filters * 2)
        self.b3 = dcgan_block_up(filters)
        self.b4 = keras.layers.Conv2DTranspose(channels, 4, 2, padding='same', activation=activation)

        if not skip:
            self.skip_connector = keras.layers.Lambda(lambda x: x[0])
        elif skip == 'residual':
            self.skip_connector = keras.layers.Add()
        else:
            self.skip_connector = keras.layers.Concatenate(axis=-1)

        self.skip = skip

    def call(self, input_tensor, training=None, **kwargs):
        if not self.skip:
            x, skip = input_tensor, [None]*3
        else:
            x, skip = input_tensor

        x = x[:, tf.newaxis, tf.newaxis, :]
        x = self.b1(x, training=training)
        x = self.b2(self.skip_connector([x, skip[2]]), training=training)
        x = self.b3(self.skip_connector([x, skip[1]]), training=training)
        x = self.b4(self.skip_connector([x, skip[0]]), training=training)

        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], 32, 32, self.channels])
