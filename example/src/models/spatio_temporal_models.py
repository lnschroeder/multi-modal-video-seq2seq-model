import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Info (1), Warning (2), Error (3) messages

import tensorflow as tf

from tensorflow import keras

from .lib.util import get_spatial_coders
from .lstm import AutoencoderLSTM, CompositeLSTM
from .lib.tdist import TimeDistributedReturnSkip, TimeDistributedRepeatSkip

from ..utils.tf.model import Model


class SrivastavaAE(Model):

    def __init__(self, params, **kwargs):
        """
        Srivastava Autoencoder LSTM
        Takes sequences of frames and reconstructs them.
        If a dict is passed it reconstructs ['labels'] instead of input.
        """
        super(SrivastavaAE, self).__init__(**kwargs)

        self.params = params
        self.simple = 'DATASET_I' not in params
        self.loss_name = params['LOSS'] if 'LOSS' in params else 'MSE'
        self.use_bce = params['USE_BCE'] if 'USE_BCE' in params else False

        spatial_encoder, spatial_decoder = get_spatial_coders(params)

        if params['SKIP_CONNECTIONS']:
            keyframe = params['SEQ_LEN'] - 1
            self._spatial_encoder = TimeDistributedReturnSkip(spatial_encoder, keyframe=keyframe)
            self._spatial_decoder = TimeDistributedRepeatSkip(spatial_decoder)
        else:
            self._spatial_encoder = keras.layers.TimeDistributed(spatial_encoder)
            self._spatial_decoder = keras.layers.TimeDistributed(spatial_decoder)

        self._lstm_autoencoder = AutoencoderLSTM(
            output_dim=params['DIM_FEATURE'],
            seq_len=params['SEQ_LEN'],
            dim_state=params['DIM_STATE'],
            lstm_count=params['LSTM_COUNT'] if 'LSTM_COUNT' in params else 1
        )

    def call(self, inputs, training=None, **kwargs):
        """
        Args:
            inputs: Tensor [batch_size, seq_len, *observation_dims]

        Returns:
            a dict containing the spatial decoding with shape (batch_size, seq_len, w, h, c)
            and the temporal encodings/representation vectors (output of the lstm encoder) with shape (batch_size, dim_state)
        """
        if self.simple:
            x = inputs
            targets = inputs
        else:
            x = inputs['inputs']
            targets = inputs['labels']

        if self.params['SKIP_CONNECTIONS']:
            spatial_encodings, skips = self._spatial_encoder(x)
            temporal_encoding, temporal_decodings = self._lstm_autoencoder(spatial_encodings)
            spatial_decodings = self._spatial_decoder(temporal_decodings, skips)
        else:
            spatial_encodings = self._spatial_encoder(x)
            temporal_encoding, temporal_decodings = self._lstm_autoencoder(spatial_encodings)
            spatial_decodings = self._spatial_decoder(temporal_decodings)

        # loss and metrics
        mse = keras.losses.MeanSquaredError()(targets, spatial_decodings)
        ssim = tf.image.ssim(targets, spatial_decodings, 1.0)
        psnr = tf.image.psnr(targets, spatial_decodings, 1.0)
        bce = None
        wbce = None

        if self.use_bce or self.loss_name == 'BCE':
            bce = keras.losses.BinaryCrossentropy()(targets, spatial_decodings)
            self.add_metric(bce, aggregation='mean', name='pixelspace/BCE')

        if self.use_bce or self.loss_name == 'WBCE':
            edge_weight = tf.reduce_sum(targets) / tf.size(targets, out_type=tf.float32)
            weights = tf.where(tf.equal(targets, 1.), 1 - edge_weight, edge_weight)
            wbce = keras.losses.BinaryCrossentropy()(targets, spatial_decodings, sample_weight=weights)
            self.add_metric(wbce, aggregation='mean', name='pixelspace/WBCE')

        if self.loss_name == 'MSE':
            loss = mse
        elif self.loss_name == 'BCE':
            loss = bce
        elif self.loss_name == 'WBCE':
            loss = wbce
        else:
            raise NotImplementedError

        self.add_metric(mse, aggregation='mean', name='pixelspace/MSE')
        self.add_metric(ssim, aggregation='mean', name='pixelspace/SSIM')
        self.add_metric(psnr, aggregation='mean', name='pixelspace/PSNR')
        self.add_loss(loss)

        return {'outputs': spatial_decodings, 'states': temporal_encoding}

    def get_config(self):
        return {
            'params': self.params
        }


class SrivastavaComposite(Model):

    def __init__(self, params, **kwargs):
        """
        Srivastava Composite LSTM
        Takes sequences of frames, reconstructs them and makes predictions for next frames.
        If a dict is passed it reconstructs and predicts ['labels'] instead of input.
        """
        super(SrivastavaComposite, self).__init__(**kwargs)

        self.params = params
        self.simple = 'DATASET_I' not in params
        self.loss_name = params['LOSS'] if 'LOSS' in params else 'MSE'
        self.use_bce = params['USE_BCE'] if 'USE_BCE' in params else False

        spatial_encoder, spatial_decoder = get_spatial_coders(params)

        if params['SKIP_CONNECTIONS']:
            keyframe = params['SEQ_LEN'] - params['PRED_LEN'] - 1
            self._spatial_encoder = TimeDistributedReturnSkip(spatial_encoder, keyframe=keyframe)
            self._spatial_decoder = TimeDistributedRepeatSkip(spatial_decoder)
        else:
            self._spatial_encoder = keras.layers.TimeDistributed(spatial_encoder)
            self._spatial_decoder = keras.layers.TimeDistributed(spatial_decoder)

        self._lstm_composite = CompositeLSTM(
            ouput_dim=params['DIM_FEATURE'],
            rec_len=params['SEQ_LEN'] - params['PRED_LEN'],
            fut_len=params['PRED_LEN'],
            dim_state=params['DIM_STATE'],
            lstm_count=params['LSTM_COUNT'] if 'LSTM_COUNT' in params else 1
        )

    def call(self, inputs, training=None, **kwargs):
        """
        Args:
            inputs: Tensor [batch_size, seq_len, *observation_dims]

        Returns:
            a dict containing the spatial decoding with shape (batch_size, seq_len, w, h, c)
            and the temporal encodings/representation vectors (output of the lstm encoder) with shape (batch_size, dim_state)
        """
        if self.simple:
            x = inputs
            targets = inputs
        else:
            x = inputs['inputs']
            targets = inputs['labels']

        if self.params['SKIP_CONNECTIONS']:
            spatial_encodings, skips = self._spatial_encoder(x)
            temporal_encoding, temporal_decodings = self._lstm_composite(spatial_encodings)
            spatial_decodings = self._spatial_decoder(temporal_decodings, skips)
        else:
            spatial_encodings = self._spatial_encoder(x)
            temporal_encoding, temporal_decodings = self._lstm_composite(spatial_encodings)
            spatial_decodings = self._spatial_decoder(temporal_decodings)

        # loss and metrics
        mse = keras.losses.MeanSquaredError()(targets, spatial_decodings)
        ssim = tf.image.ssim(targets, spatial_decodings, 1.0)
        psnr = tf.image.psnr(targets, spatial_decodings, 1.0)
        bce = None
        wbce = None

        if self.use_bce or self.loss_name == 'BCE':
            bce = keras.losses.BinaryCrossentropy()(targets, spatial_decodings)
            self.add_metric(bce, aggregation='mean', name='pixelspace/BCE')

        if self.use_bce or self.loss_name == 'WBCE':
            edge_weight = tf.reduce_sum(targets) / tf.size(targets, out_type=tf.float32)
            weights = tf.where(tf.equal(targets, 1.), 1 - edge_weight, edge_weight)
            wbce = keras.losses.BinaryCrossentropy()(targets, spatial_decodings, sample_weight=weights)
            self.add_metric(wbce, aggregation='mean', name='pixelspace/WBCE')

        if self.loss_name == 'MSE':
            loss = mse
        elif self.loss_name == 'BCE':
            loss = bce
        elif self.loss_name == 'WBCE':
            loss = wbce
        else:
            raise NotImplementedError

        self.add_metric(mse, aggregation='mean', name='pixelspace/MSE')
        self.add_metric(ssim, aggregation='mean', name='pixelspace/SSIM')
        self.add_metric(psnr, aggregation='mean', name='pixelspace/PSNR')
        self.add_loss(loss)

        return {'outputs': spatial_decodings, 'states': temporal_encoding}

    def get_config(self):
        return {
            'params': self.params
        }
