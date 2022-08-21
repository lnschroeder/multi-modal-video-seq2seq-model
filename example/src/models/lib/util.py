import tensorflow as tf
import tensorflow_hub as hub

from .dcgan import dcgan_encoder_32, dcgan_decoder_32
from .dcgan import dcgan_encoder_64, dcgan_decoder_64
from .dcgan import dcgan_encoder_128, dcgan_decoder_128
from .vgg import vgg_encoder_64, vgg_decoder_64
from .vgg import vgg_encoder_128, vgg_decoder_128
from .resnet import resnet_encoder, resnet_decoder

from .. import segmentation_models as sm


def get_spatial_coders(params):
    """
    Returns spatial encoder and decoder specified in the params dict.
    Freezes pretrained coders.  TODO

    Args:
        params: a dict containing the necessary keys to identify the spatial coders

    Returns:
        a tuple containing the encoder and decoder
    """

    activation = params['FEATURE_ACTIVATION'] if 'FEATURE_ACTIVATION' in params else 'linear'
    skip_connections = params['SKIP_CONNECTIONS'] if 'SKIP_CONNECTIONS' in params else False
    return_skips = True if skip_connections == 'residual' or skip_connections else False
    channels_out = params['CHANNELS_OUT'] if 'CHANNELS_OUT' in params else params['DIM_OBSERVATION'][2]
    encoder_weights = params['ENC_WEIGHTS'] if 'ENC_WEIGHTS' in params and params['ENC_WEIGHTS'] != 'None' else None

    if params['SPATIAL_CODER'] == 'dcgan':
        if params['DIM_OBSERVATION'][0] == 32:
            encoder = dcgan_encoder_32(dim_out=params['DIM_FEATURE'], activation=activation, return_skips=return_skips)
            decoder = dcgan_decoder_32(channels=channels_out, activation='sigmoid', skip=skip_connections)
        elif params['DIM_OBSERVATION'][0] == 64:
            encoder = dcgan_encoder_64(dim_out=params['DIM_FEATURE'], activation=activation, return_skips=return_skips)
            decoder = dcgan_decoder_64(channels=channels_out, activation='sigmoid', skip=skip_connections)
        elif params['DIM_OBSERVATION'][0] == 128:
            encoder = dcgan_encoder_128(dim_out=params['DIM_FEATURE'], activation=activation, return_skips=return_skips)
            decoder = dcgan_decoder_128(channels=channels_out, activation='sigmoid', skip=skip_connections)
        else:
            raise NotImplementedError

    elif params['SPATIAL_CODER'] == 'vgg':
        if params['DIM_OBSERVATION'][0] == 64:
            encoder = vgg_encoder_64(dim_out=params['DIM_FEATURE'], activation=activation, return_skips=return_skips)
            decoder = vgg_decoder_64(channels=channels_out, activation='sigmoid', skip=skip_connections)
        elif params['DIM_OBSERVATION'][0] == 128:
            encoder = vgg_encoder_128(dim_out=params['DIM_FEATURE'], activation=activation, return_skips=return_skips)
            decoder = vgg_decoder_128(channels=channels_out, activation='sigmoid', skip=skip_connections)
        else:
            raise NotImplementedError

    elif params['SPATIAL_CODER'] == 'resnet':

        if params['DIM_OBSERVATION'][0] == 64:
            n_sampling = 4
        elif params['DIM_OBSERVATION'][0] == 128:
            n_sampling = 5
        elif params['DIM_OBSERVATION'][0] == 224:
            n_sampling = 5
        else:
            raise NotImplementedError

        encoder = resnet_encoder(
            dim_out=params['DIM_FEATURE'],
            dim_in=params['DIM_OBSERVATION'][0],
            channels=params['DIM_OBSERVATION'][2],
            activation=activation,
            n_sampling=n_sampling,
            return_skips=return_skips
        )
        decoder = resnet_decoder(
            dim_in=params['DIM_FEATURE'],
            dim_out=params['DIM_OBSERVATION'][0],  # TODO unused parameter
            channels=channels_out,
            activation='sigmoid',
            n_sampling=n_sampling,
            skip=skip_connections
        )

    elif params['SPATIAL_CODER'] == 'linknet':
        sm.set_framework('tf.keras')

        # define model
        encoder, decoder, _ = sm.Linknet(
            dim_encoding=params['DIM_FEATURE'],
            activation='sigmoid',
            backbone_name=params['BACKBONE'],
            input_shape=params['DIM_OBSERVATION'],
            classes=channels_out,
            encoder_weights=encoder_weights
        )
    elif params['SPATIAL_CODER'] == 'unet':
        sm.set_framework('tf.keras')

        # define model
        encoder, decoder, _ = sm.Unet(
            dim_encoding=params['DIM_FEATURE'],
            activation='sigmoid',
            # https://github.com/qubvel/classification_models/blob/master/classification_models/models_factory.py#L10-L60
            backbone_name=params['BACKBONE'],
            input_shape=params['DIM_OBSERVATION'],
            classes=channels_out,
            # https://github.com/qubvel/classification_models/blob/master/classification_models/weights.py#L40-L390
            encoder_weights=encoder_weights
        )
    else:
        raise NotImplementedError

    return encoder, decoder
