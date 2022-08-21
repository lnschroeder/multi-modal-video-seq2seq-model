import tensorflow as tf
from tensorflow import keras


def get_activation_func(activation, dtype=None):

    if activation == 'elu':
        return keras.layers.ELU(alpha=1.0)
    elif activation == 'leaky_relu':
        return keras.layers.LeakyReLU(alpha=0.3, dtype=dtype)
    else:
        return keras.layers.Activation(activation, dtype=dtype)


class conv2d_with_padding(keras.layers.Layer):

    def __init__(self, filters, kernel_size=3, strides=1, name='resnet_conv', **kwargs):
        super(conv2d_with_padding, self).__init__(name=name, **kwargs)

        # Do adjusted padding when strides > 1, if not let the conv layer do the padding
        self.padding = self._adjusted_padding if strides > 1 else (lambda x : x)

        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'),
            use_bias=False
        )

        self.ksize = kernel_size

    def _adjusted_padding(self, x):
        """ Pads input tensor with pad size ksize - 1.

        Args:
        x: Tensor [batch_size, height, width, channels]
        ksize: Integer
          The kernel used in conv and max_pool layers.

        Returns:
        x: Tensor [batch_size, height+pad_s, width+pad_e, channels]
          Padded input tensor.
        """
        pad = self.ksize - 1
        pad_s = pad // 2
        pad_e = pad - pad_s
        return tf.pad(x, [[0, 0], [pad_s, pad_e], [pad_s, pad_e], [0, 0]])

    def call(self, input_tensor, training=None):
        return self.conv(self.padding(input_tensor), training=training)


class conv_block(keras.layers.Layer):

    def __init__(
            self, filters, ksize=3, strides=1, padding='same',
            activation='leaky_relu', normalization=True,
            dtype=None, name='conv_block', **kwargs
    ):
        super(conv_block, self).__init__(name=name, **kwargs)
        self._norm = normalization

        if self._norm:
            self.norm = keras.layers.BatchNormalization(dtype=dtype)

        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=ksize,
            strides=strides,
            use_bias=(not self._norm),
            padding=padding,
            dtype=dtype
        )
        self.activation = get_activation_func(activation, dtype=dtype)


    def call(self, input_tensor, training=None):
        x = self.conv(input_tensor, training=training)
        if self._norm:
            x = self.norm(x, training=training)
        return self.activation(x)


class conv_block_up(keras.layers.Layer):

    def __init__(
            self, filters, ksize=3, strides=2, padding='same',
            activation='leaky_relu', dropout=0.0,
            dtype=None, name='upsample_block', **kwargs
    ):
        super(conv_block_up, self).__init__(name=name, **kwargs)
        self.norm = keras.layers.BatchNormalization(dtype=dtype)
        self.conv = keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=ksize,
            strides=strides,
            padding=padding,
            use_bias=False,
            dtype=dtype
        )
        self.dropout = keras.layers.Dropout(0.0)
        self.activation = get_activation_func(activation, dtype=dtype)

    def call(self, input_tensor, training=None):
        x = self.conv(input_tensor, training=training)
        x = self.norm(x, training=training)
        x = self.dropout(x, training=training)
        return self.activation(x)


class resnet_block(keras.layers.Layer):

    def __init__(
            self, filters, ksize=3, strides=1, ksize_proj=1, activation='leaky_relu',
            output_filters=None, output_ksize=None, output_strides=1, output_padding='same',
            dtype=None, name='resnet_block', **kwargs
    ):
        super(resnet_block, self).__init__(name=name, **kwargs)

        self.inp_norm = keras.layers.BatchNormalization(dtype=dtype)
        self.out_norm = keras.layers.BatchNormalization(dtype=dtype)

        self.inp_conv = conv2d_with_padding(
            filters=filters,
            kernel_size=ksize,
            strides=strides,
            dtype=dtype
        )

        self.out_conv = keras.layers.Conv2D(
            filters=(output_filters or filters),
            kernel_size=(output_ksize or ksize),
            strides=output_strides,
            padding=output_padding,
            use_bias=False,
            dtype=dtype
        )

        self._do_projection = ksize_proj > 0
        if self._do_projection:
            self.projection = keras.layers.Conv2D(
                filters=(output_filters or filters),
                kernel_size=ksize_proj,
                strides=strides*output_strides,
                dtype=dtype
            )
            self.pro_norm = keras.layers.BatchNormalization(dtype=dtype)


        self.activation = get_activation_func(activation, dtype=dtype)

    def call(self, input_tensor, training=None):
        shortcut = input_tensor

        if self._do_projection:
            shortcut = self.projection(shortcut, training=training)
            shortcut = self.pro_norm(shortcut, training=training)

        x = self.inp_conv(input_tensor, training=training)
        x = self.inp_norm(x, training=training)
        x = self.activation(x)

        x = self.out_conv(x, training=training)
        x = self.out_norm(x, training=training)

        return self.activation(shortcut + x)
