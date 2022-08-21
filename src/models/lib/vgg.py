"""
Author of file: Manuel Woellhaf
"""
import tensorflow as tf
from tensorflow import keras


class vgg_block(keras.layers.Layer):

    def __init__(self, filters, ksize=3, activation='leaky_relu', padding='same', use_bias=False, name='vgg_block', **kwargs):
        super(vgg_block, self).__init__(name=name, **kwargs)
        self.conv = keras.layers.Conv2D(filters, ksize, 1, padding=padding, use_bias=use_bias)
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


class vgg_block_up(keras.layers.Layer):

    def __init__(self, filters, ksize=4, activation='leaky_relu', padding='same', use_bias=False, name='vgg_block_up', **kwargs):
        super(vgg_block_up, self).__init__(name=name, **kwargs)
        self.conv = keras.layers.Conv2DTranspose(filters, ksize, 1, padding=padding, use_bias=use_bias)
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


class vgg_encoder_128(keras.Model):
    """
    ~ xx Mio trainable parameters.
    """
    def __init__(self, dim_out, activation='linear', return_skips=False, name='vgg_encoder', **kwargs):
        super(vgg_encoder_128, self).__init__(name=name, **kwargs)
        self.dim_out = dim_out

        # 128 x 128
        self.b11 = vgg_block( 64)
        self.b12 = vgg_block( 64)
        # 64 x 64
        self.b21 = vgg_block(128)
        self.b22 = vgg_block(128)
        # 32 x 32
        self.b31 = vgg_block(256)
        self.b32 = vgg_block(256)
        self.b33 = vgg_block(256)
        # 16 x 16
        self.b41 = vgg_block(512)
        self.b42 = vgg_block(512)
        self.b43 = vgg_block(512)
        # 8 x 8
        self.b51 = vgg_block(512)
        self.b52 = vgg_block(512)
        self.b53 = vgg_block(512)
        # 4 x 4
        self.out = vgg_block(dim_out, ksize=4, activation=activation, padding='valid')
        self.mp = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        self.return_skips = return_skips

    def call(self, input_tensor, training=None):
        x = self.b11(input_tensor, training=training)
        h1 = self.b12(x, training=training)

        x = self.b21(self.mp(h1), training=training)
        h2 = self.b22(x, training=training)

        x = self.b31(self.mp(h2), training=training)
        x = self.b32(x, training=training)
        h3 = self.b33(x, training=training)

        x = self.b41(self.mp(h3), training=training)
        x = self.b42(x, training=training)
        h4 = self.b43(x, training=training)

        x = self.b51(self.mp(h4), training=training)
        x = self.b52(x, training=training)
        h5 = self.b53(x, training=training)

        h6 = self.out(self.mp(h5), training=training)

        if self.return_skips:
            return h6[:, 0, 0, :], [h1, h2, h3, h4, h5]
        else:
            return h6[:, 0, 0, :]

    def compute_output_shape(self, input_shape):
        if self.return_skips:
            return [
                tf.TensorShape([input_shape[0], self.dim_out]),
                [
                    tf.TensorShape([input_shape[0], 128, 128,  32]),
                    tf.TensorShape([input_shape[0],  64,  64,  64]),
                    tf.TensorShape([input_shape[0],  32,  32, 128]),
                    tf.TensorShape([input_shape[0],  16,  16, 256]),
                    tf.TensorShape([input_shape[0],   8,   8, 512])
                ]
            ]
        else:
            return tf.TensorShape([input_shape[0], self.dim_out])


class vgg_decoder_128(keras.Model):
    """
    ~ xx Mio trainable parameters for skip='residual' and 20 Mio for skip=True.
    """
    def __init__(self, channels=3, activation='linear', skip=False, name='vgg_decoder', **kwargs):
        super(vgg_decoder_128, self).__init__(name=name, **kwargs)
        self.channels = channels

        # 4 x 4
        self.b11 = vgg_block_up(512, ksize=4, padding='valid')
        # 8 x 8
        self.b21 = vgg_block(512)
        self.b22 = vgg_block(512)
        self.b23 = vgg_block(512)
        # 16 x 16
        self.b31 = vgg_block(512)
        self.b32 = vgg_block(512)
        self.b33 = vgg_block(256)
        # 32 x 32
        self.b41 = vgg_block(256)
        self.b42 = vgg_block(256)
        self.b43 = vgg_block(128)
        # 64 x 64
        self.b51 = vgg_block(128)
        self.b52 = vgg_block( 64)
        # 128 x 128
        self.out = keras.layers.Conv2DTranspose(channels, 3, 1, padding='same', activation=activation)

        self.up = keras.layers.UpSampling2D(size=(2, 2))

        if not skip:
            self.skip_connector = keras.layers.Lambda(lambda x: x[0])
        elif skip == 'residual':
            self.skip_connector = keras.layers.Add()
        else:
            self.skip_connector = keras.layers.Concatenate(axis=-1)

        self.skip = skip

    def call(self, input_tensors, training=None):

        if not self.skip:
            x, skip = input_tensors, [None]*5
        else:
            x, skip = input_tensors

        x = x[:, tf.newaxis, tf.newaxis, :]

        x = self.b11(x, training=training)
        x = self.up(x)

        x = self.b21(self.skip_connector([x, skip[4]]), training=training)
        x = self.b22(x, training=training)
        x = self.b23(x, training=training)
        x = self.up(x)

        x = self.b31(self.skip_connector([x, skip[3]]), training=training)
        x = self.b32(x, training=training)
        x = self.b33(x, training=training)
        x = self.up(x)

        x = self.b41(self.skip_connector([x, skip[2]]), training=training)
        x = self.b42(x, training=training)
        x = self.b43(x, training=training)
        x = self.up(x)

        x = self.b51(self.skip_connector([x, skip[1]]), training=training)
        x = self.b52(x, training=training)
        x = self.up(x)

        x = self.out(self.skip_connector([x, skip[0]]))

        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], 128, 128, self.channels])


class vgg_encoder_64(keras.Model):

    def __init__(self, dim_out, activation='linear', return_skips=False, name='vgg_encoder', **kwargs):
        super(vgg_encoder_64, self).__init__(name=name, **kwargs)
        self.dim_out = dim_out

        # 64 x 64
        self.b11 = vgg_block(64)
        self.b12 = vgg_block(64)
        # 32 x 32
        self.b21 = vgg_block(128)
        self.b22 = vgg_block(128)
        # 16 x 16
        self.b31 = vgg_block(256)
        self.b32 = vgg_block(256)
        self.b33 = vgg_block(256)
        # 8 x 8
        self.b41 = vgg_block(512)
        self.b42 = vgg_block(512)
        self.b43 = vgg_block(512)
        # 4 x 4
        self.out = vgg_block(dim_out, ksize=4, activation=activation, padding='valid')
        self.mp = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        self.return_skips = return_skips

    def call(self, input_tensor, training=None):
        x = self.b11(input_tensor, training=training)
        h1 = self.b12(x, training=training)

        x = self.b21(self.mp(h1), training=training)
        h2 = self.b22(x, training=training)

        x = self.b31(self.mp(h2), training=training)
        x = self.b32(x, training=training)
        h3 = self.b33(x, training=training)

        x = self.b41(self.mp(h3), training=training)
        x = self.b42(x, training=training)
        h4 = self.b43(x, training=training)

        h5 = self.out(self.mp(h4), training=training)

        if self.return_skips:
            return h5[:, 0, 0, :], [h1, h2, h3, h4]
        else:
            return h5[:, 0, 0, :]

    def compute_output_shape(self, input_shape):
        if self.return_skips:
            return [
                tf.TensorShape([input_shape[0], self.dim_out]),
                [
                    tf.TensorShape([input_shape[0], 64, 64,  64]),
                    tf.TensorShape([input_shape[0], 32, 32, 128]),
                    tf.TensorShape([input_shape[0], 16, 16, 256]),
                    tf.TensorShape([input_shape[0],  8,  8, 512])
                ]
            ]
        else:
            return tf.TensorShape([input_shape[0], self.dim_out])


class vgg_decoder_64(keras.Model):

    def __init__(self, channels=3, activation='linear', skip=False, name='vgg_decoder', **kwargs):
        super(vgg_decoder_64, self).__init__(name=name, **kwargs)
        self.channels = channels

        # 4 x 4
        self.b11 = vgg_block_up(512, ksize=4, padding='valid', use_bias=True)
        # 8 x 8
        self.b21 = vgg_block(512)
        self.b22 = vgg_block(512)
        self.b23 = vgg_block(256)
        # 16 x 16
        self.b31 = vgg_block(256)
        self.b32 = vgg_block(256)
        self.b33 = vgg_block(128)
        # 32 x 32
        self.b41 = vgg_block(128)
        self.b42 = vgg_block(64)
        # 64 x 64
        # self.b51 = vgg_block(64)
        self.out = keras.layers.Conv2DTranspose(channels, 3, 1, padding='same', activation=activation)

        self.up = keras.layers.UpSampling2D(size=(2, 2))

        if not skip:
            self.skip_connector = keras.layers.Lambda(lambda x: x[0])
        elif skip == 'residual':
            self.skip_connector = keras.layers.Add()
        else:
            self.skip_connector = keras.layers.Concatenate(axis=-1)

        self.skip = skip

    def call(self, input_tensors, training=None):

        if not self.skip:
            x, skip = input_tensors, [None]*4
        else:
            x, skip = input_tensors

        x = x[:, tf.newaxis, tf.newaxis, :]

        x = self.b11(x, training=training)
        x = self.up(x)

        x = self.b21(self.skip_connector([x, skip[3]]), training=training)
        x = self.b22(x, training=training)
        x = self.b23(x, training=training)
        x = self.up(x)

        x = self.b31(self.skip_connector([x, skip[2]]), training=training)
        x = self.b32(x, training=training)
        x = self.b33(x, training=training)
        x = self.up(x)

        x = self.b41(self.skip_connector([x, skip[1]]), training=training)
        x = self.b42(x, training=training)

        x = self.up(x)
        x = self.out(self.skip_connector([x, skip[0]]))

        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], 64, 64, self.channels])
