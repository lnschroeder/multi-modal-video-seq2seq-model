"""
Author of file: Manuel Woellhaf
"""
import tensorflow as tf
from tensorflow import keras

from .layers import conv2d_with_padding, conv_block, resnet_block, conv_block_up, get_activation_func


def resnet_encoder(
        dim_out, dim_in, channels=3,
        n_filters=32, n_blocks=2, n_sampling=4,
        activation='linear', return_skips=False, name='resnet_encoder'
):

    # define layers ------------------------------------------------------------
    inp_layer = conv_block(n_filters, ksize=7, normalization=False, name=name+'/conv_block_0')

    sampling_layers = []
    resnet_blocks = []

    for i in range(n_sampling):

        filters = n_filters*2**min(i+1, n_sampling-1)

        sampling_layers.append(
            conv_block(filters, strides=2, name=name+'/conv_block_'+str(i+1))
        )

        resnet_blocks.append(
            [resnet_block(filters, name=name+'/resnet_block_'+str(i+1)+str(j)) for j in range(n_blocks)]
        )


    out_layer = conv_block(dim_out, ksize=int(dim_in//2**n_sampling), activation=activation, padding='valid',
                           name=name+'/conv_block_out')

    # define model -------------------------------------------------------------
    inputs = tf.keras.layers.Input(shape=[dim_in, dim_in, channels])
    skips = []


    x = inp_layer(inputs)

    for i in range(n_sampling):
        x = sampling_layers[i](x)
        skips.append(x)
        for j in range(n_blocks):
            x = resnet_blocks[i][j](x)

    skips.reverse()
    x = out_layer(x)[:, 0, 0, :]

    if return_skips:
        return tf.keras.Model(inputs=inputs, outputs=[x, skips])
    else:
        return tf.keras.Model(inputs=inputs, outputs=x)


def resnet_decoder(
        dim_in, dim_out=None, channels=3,
        n_filters=32, n_blocks=2, n_sampling=4,
        activation='linear', skip=False, name='resnet_decoder'
):
    # define layers ------------------------------------------------------------
    inp_layer = conv_block_up(n_filters*2**(n_sampling-1), ksize=4, strides=1, padding='valid', name=name+'/conv_block_0')

    sampling_layers = []
    resnet_blocks = []
    filters = [n_filters*2**min(n_sampling-i, n_sampling-1) for i in range(n_sampling)]

    for i in range(n_sampling-1):

        sampling_layers.append(
            conv_block_up(filters[i+1], name=name+'/conv_block_'+str(i+1))
        )

        resnet_blocks.append(
            [resnet_block(filters[i], name=name+'/resnet_block_'+str(i+1)+str(j)) for j in range(n_blocks)]
        )

    # we don't want to have resnet blocks in the full resolution, too expensive
    sampling_layers.append(conv_block_up(n_filters, name=name+'/conv_block_'+str(n_sampling)))

    out_layer = conv_block(channels, ksize=7, activation=activation, normalization=False,
                           name=name+'/resnet_block_out')

    # define model -------------------------------------------------------------
    inputs = tf.keras.layers.Input(shape=[dim_in])
    skips = [tf.keras.layers.Input(shape=[None, None, filters[i]]) for i in range(n_sampling)]

    x = inputs[:, tf.newaxis, tf.newaxis, :]
    x = inp_layer(x)

    for i in range(n_sampling-1):

        if skip:
            x = x + skips[i]

        for j in range(n_blocks):
            x = resnet_blocks[i][j](x)

        x = sampling_layers[i](x)

    x = sampling_layers[-1](x)
    x = out_layer(x)

    if skips:
        return tf.keras.Model(inputs=[inputs, skips], outputs=x)
    else:
        return tf.keras.Model(inputs=inputs, outputs=x)
