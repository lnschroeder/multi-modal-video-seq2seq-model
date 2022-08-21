import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense


class LSTMEncoder(keras.layers.Layer):

    def __init__(
            self,
            latent_dim,
            lstm_count=1,
            name='lstm_encoder',
            **kwargs
    ):
        """
        Args:
            latent_dim: dimension of the lstm
            lstm_count: amount of lstm layers
        """
        super().__init__(name=name, **kwargs)

        self.latent_dim = latent_dim
        self.lstm_count = lstm_count

        if lstm_count == 1:
            self._lstm_1 = LSTM(
                units=latent_dim,
                return_sequences=False,
                return_state=True
            )
        elif lstm_count == 2:
            self._lstm_1 = LSTM(
                units=latent_dim,
                return_sequences=True,
                return_state=False
            )
            self._lstm_2 = LSTM(
                units=latent_dim,
                return_sequences=False,
                return_state=True
            )
        else:
            raise NotImplementedError

    def call(self, inputs, training=None):
        """
        Args:
            inputs: input for Encoder has shape (batch_size, seq_len, dim_feature)
            training: training argument

        Returns:
            returns state_h, and states.
                state_h has shape (batch_size, latent_dim).
                states is [states_h, states_c] and each of them have shape (batch_size, latent_dim)
        """
        if self.lstm_count == 1:
            state_h, *states = self._lstm_1(inputs)
        else:
            states_h = self._lstm_1(inputs)
            state_h, *states = self._lstm_2(states_h)

        return state_h, states

    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'lstm_count': self.lstm_count,
            'name': self.name
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class LSTMDecoder(keras.layers.Layer):

    def __init__(
            self,
            output_dim,
            seq_len,
            latent_dim,
            lstm_count=1,
            reverse=False,
            name='lstm_decoder',
            **kwargs
    ):
        """
        Args:
            output_dim: dimension of output vector
            seq_len: length of the output sequence
            latent_dim: dimension of the lstm
            lstm_count: amount of lstm layers
            reverse: if True the last timestep gets predicted first
        """
        super().__init__(name=name, **kwargs)

        self.output_dim = output_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.reverse = reverse
        self.lstm_count = lstm_count
        if self.lstm_count == 1:
            self._lstm_1 = LSTM(
                units=latent_dim,
                return_sequences=True,
                return_state=True
            )
        else:
            self._lstm_1 = LSTM(
                units=latent_dim,
                return_sequences=True,
                return_state=False
            )
            self._lstm_2 = LSTM(
                units=latent_dim,
                return_sequences=True,
                return_state=True
            )

        self._dense = Dense(output_dim, activation='softmax')

    def call(self, inputs, training=False, **kwargs):
        """
        Args:
            inputs: input of Decoder.
                When training=True inputs is [initial_states, targets]; else inputs is [initial_states].
                initial_states is [states_h, states_c] and each of them have shape (batch_size, latent_dim).
                targets has shape (batch_size, seq_len, output_dim)
            training: training argument

        Returns:
            output_tensors has shape (batch_size, seq_len, output_dim)
        """
        if training:
            # conditional decoder
            initial_states, targets = inputs

            if self.reverse:
                targets = tf.reverse(targets, axis=[1])

            input_tensors = tf.pad(targets[:, :-1], [[0, 0], [1, 0], [0, 0]])  # insert dummy tensor at first element

            if self.lstm_count == 1:
                state_h, *_ = self._lstm_1(input_tensors, initial_state=initial_states)
            else:
                states_h = self._lstm_1(input_tensors, initial_state=initial_states)
                state_h, *_ = self._lstm_2(states_h)

            output_tensors = self._dense(state_h)
        else:
            # unconditional decoder
            initial_states = inputs[0]
            batch_size = tf.shape(initial_states[0])[0]
            input_tensor = tf.zeros([batch_size, 1, self.output_dim])

            output_tensors = []
            for _ in range(self.seq_len):
                if self.lstm_count == 1:
                    state_h, *initial_states = self._lstm_1(input_tensor, initial_state=initial_states)
                else:
                    states_h = self._lstm_1(input_tensor, initial_state=initial_states)
                    state_h, *initial_states = self._lstm_2(states_h)
                output_tensor = self._dense(state_h)
                output_tensors.append(output_tensor)
                input_tensor = output_tensor
            output_tensors = tf.concat(output_tensors, axis=1)

        if self.reverse:
            output_tensors = tf.reverse(output_tensors, axis=[1])

        return output_tensors

    def get_config(self):
        return {
            'output_dim': self.output_dim,
            'seq_len': self.seq_len,
            'latent_dim': self.latent_dim,
            'lstm_count': self.lstm_count,
            'reverse': self.reverse,
            'name': self.name
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
