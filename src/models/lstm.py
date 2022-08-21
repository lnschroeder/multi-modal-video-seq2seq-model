import tensorflow as tf

from tensorflow import keras

from .encdeclstm import LSTMEncoder, LSTMDecoder


class AutoencoderLSTM(keras.layers.Layer):

    def __init__(
            self,
            output_dim,
            seq_len,
            dim_state,
            lstm_count,
            name='autoencoder_lstm',
            **kwargs
    ):
        """
        Srivastava's autoencoder LSTM. Generates output in reversed order
        (last input gets predicted first).

        See https://arxiv.org/pdf/1502.04681.pdf

        Args:
            output_dim: dimension of output tensor
            seq_len: amount of frames
            dim_state: representation state dimension
            lstm_count: amount of lstm layers in encoder and decoder
        """
        super().__init__(name=name, **kwargs)

        self.output_dim = output_dim
        self.seq_len = seq_len
        self.dim_state = dim_state
        self.lstm_count = lstm_count

        self._lstm_encoder = LSTMEncoder(
            latent_dim=dim_state,
            lstm_count=lstm_count,
        )
        self._lstm_decoder = LSTMDecoder(
            output_dim=output_dim,
            seq_len=seq_len,
            latent_dim=dim_state,
            lstm_count=lstm_count,
            reverse=True
        )

    def call(self, inputs, training=False, **kwargs):
        """
        Args:
            inputs: inputs for AutoencoderLSTM. Has shape (batch_size, seq_len, dim_state)
            training: training argument

        Returns:
           returns lstm encodings with shape (batch_size, dim_state)
           and inputs sequence with shape (batch_size, seq_len, output_dim)
        """

        # temporal encoding
        lstm_encoding, states = self._lstm_encoder(inputs)

        # temporal decoding
        if training:
            lstm_decodings = self._lstm_decoder([states, inputs])
        else:
            lstm_decodings = self._lstm_decoder([states])

        return lstm_encoding, lstm_decodings

    def get_config(self):
        return {
            'output_dim': self.output_dim,
            'seq_len': self.seq_len,
            'dim_state': self.dim_state,
            'lstm_count': self.lstm_count,
            'name': self.name,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class CompositeLSTM(tf.keras.layers.Layer):

    def __init__(
            self,
            ouput_dim,
            rec_len,
            fut_len,
            dim_state,
            lstm_count,
            name='composite_lstm',
            **kwargs
    ):
        """
        Srivastava's composite LSTM. Takes rec_len input tensors, encodes them,
        reconstructs them and predicts fut_len. Reconstruction decoder generates
        output in reversed order (last input gets predicted first).

        See https://arxiv.org/pdf/1502.04681.pdf

        Args:
            ouput_dim: dimension of output tensor
            rec_len: amount of frames for reconstruction decoder
            fut_len: amount of frames for future prediction decoder
            dim_state: representation state dimension
            lstm_count: amount of lstm layers in encoder and decoder
        """
        super().__init__(name=name, **kwargs)

        self.output_dim = ouput_dim
        self.rec_len = rec_len
        self.fut_len = fut_len
        self.dim_state = dim_state
        self.lstm_count = lstm_count

        self._lstm_encoder = LSTMEncoder(
            latent_dim=dim_state,
            lstm_count=lstm_count,
        )
        self._lstm_decoder_rec = LSTMDecoder(
            output_dim=ouput_dim,
            seq_len=rec_len,
            latent_dim=dim_state,
            lstm_count=lstm_count,
            reverse=True
        )
        self._lstm_decoder_fut = LSTMDecoder(
            output_dim=ouput_dim,
            seq_len=fut_len,
            latent_dim=dim_state,
            lstm_count=lstm_count,
            reverse=False
        )

    def call(self, inputs, training=False, **kwargs):
        """
        Args:
            inputs: inputs for CompositeLSTM. Has shape (batch_size, seq_len, dim_state)
            training: training argument

        Returns:
            returns lstm encodings with shape (batch_size, dim_state)
            and sequence with shape (batch_size, rec_len + fut_len, output_dim)
        """
        # temporal encoding
        target_rec = inputs[:, :self.rec_len]
        target_fut = inputs[:, self.rec_len:]

        lstm_encoding, states = self._lstm_encoder(target_rec)

        # temporal decoding
        if training:
            lstm_decodings_rec = self._lstm_decoder_rec([states, target_rec])
            lstm_decodings_fut = self._lstm_decoder_fut([states, target_fut])
        else:
            lstm_decodings_rec = self._lstm_decoder_rec([states])
            lstm_decodings_fut = self._lstm_decoder_fut([states])

        lstm_decodings = tf.concat((lstm_decodings_rec, lstm_decodings_fut), axis=1)

        return lstm_encoding, lstm_decodings

    def get_config(self):
        return {
            'output_dim': self.output_dim,
            'rec_len': self.rec_len,
            'fut_len': self.fut_len,
            'dim_state': self.dim_state,
            'lstm_count': self.lstm_count,
            'name': self.name
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
