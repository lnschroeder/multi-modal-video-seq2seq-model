"""
MNIST is a dataset of handwritten digits available on http://yann.lecun.com/exdb/mnist/.
Moving MNIST is introduced in Unsupervised Learning of Video Representations Using LSTMs by
Nitish Srivastava, Elman Mansimov and Ruslan Salakhutdinov (https://arxiv.org/abs/1502.04681).
Stochastic Moving MNIST is introduces in Stochastic Video Generation with a Learned Prior by
Emily Denton and Rob Fergus (https://arxiv.org/abs/1802.07687)

This code is based on the script by Tencia Lee (https://gist.github.com/tencia/afb129122a64bde3bd0c).
"""
import math

import numpy as np

from .mnist import mnist


def samples_to_batch(samples):

    samples = samples if type(samples) == list else [samples]

    batch = (
        np.stack([s[0] for s in samples], axis=0),
        {
            'decoding': np.stack([s[1]['decoding'] for s in samples], axis=0),
            'prediction': np.stack([s[1]['prediction'] for s in samples], axis=0)
        }
    )

    return batch


class moving_mnist:

    def __init__(
            self,
            mnist_path,
            shape=(64, 64),
            seq_len=20,
            digits=2,
            colored=False,
            train=None,
            labels='categorical',
            stochastic=False,
            seed=None
    ):
        """
        Args:
          train: boolean (default None)
           Full set if None, training set if True and test set if False.
          colored: boolean
        """
        shape = shape if type(shape) != int else (shape, shape)

        self._seq_len = seq_len
        self._digits = digits
        self._colored = colored
        self._labels = labels
        self._stochastic = stochastic

        if self._colored:
            self.shape = [(self._seq_len, shape[0], shape[1], 3), self._digits]
        else:
            self.shape = [(self._seq_len, shape[0], shape[1], 1), self._digits]

        data = mnist(mnist_path)
        if train is None:
            self._imgs, self._lbls = data.full(flatten=False, integral=True)
        elif train:
            self._imgs, self._lbls = data.train(flatten=False, integral=True)
        else:
            self._imgs, self._lbls = data.test(flatten=False, integral=True)
        self._imgs = np.squeeze(self._imgs)

        self.random = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __len__(self):
        return self._imgs.shape[0]//self._digits

    def __next__(self):
        sequence = np.empty(self.shape[0], dtype=np.uint16)

        def sample_velocity():
            angle = np.pi * (self.random.rand(self._digits)*2 - 1)
            speed = self.random.randint(5, size=self._digits)+2
            return np.array([(s*math.cos(a), s*math.sin(a)) for a, s in zip(angle, speed)])

        limits = np.array(self.shape[0][1:3]) - self._imgs.shape[1:3]
        indices = self.random.randint(self._imgs.shape[0], size=self._digits)
        velocity = sample_velocity()
        position = (self.random.rand(self._digits, 2)*limits)
        colors = [self.random.randint(low=0, high=255, size=3) for i in range(self._digits)]

        for frame_idx in range(self._seq_len):

            # generate digits
            digits = np.zeros((self.shape[0][1], self.shape[0][2], self._digits), dtype=np.uint8)
            for i in range(self._digits):
                s, e = position[i].astype(np.uint8), position[i].astype(np.uint8) + self._imgs.shape[1:3]
                digits[s[0]:e[0], s[1]:e[1], i] = self._imgs[indices[i]]

            # update position
            new_pos = position + velocity
            bounce = (new_pos < 0) + (limits <= new_pos)
            if np.any(bounce):
                if self._stochastic:
                    velocity = sample_velocity()
                    new_pos = position + velocity
                    bounce = (new_pos < 0) + (limits <= new_pos)

                velocity *= (np.ones(velocity.shape)-2*bounce)

            position = position + velocity

            # generate frame
            frame = []
            for i in range(self._digits):
                if self._colored:
                    frame.append(digits[..., i:i+1]*colors[i])
                else:
                    frame.append(digits[..., i:i+1])

            # add frame to sequence
            sequence[frame_idx] = sum(frame)

        sequence = sequence/255 if self._colored else sequence
        sequence = sequence.clip(0, 255).astype(np.float32)/255.0

        if self._labels is None:
            return sequence

        if self._labels == 'categorical':
            return (
                sequence,
                np.array([self._lbls[indices[i]] for i in range(self._digits)], dtype=np.int32)
            )

        if self._labels == 'duplicate':
            return (sequence, sequence)

        if self._labels == 'composit':
            return (
                sequence[:self._seq_len//2],
                {
                    'decoding': np.flip(sequence[:self._seq_len//2], axis=0),
                    'prediction': sequence[self._seq_len//2:]
                }
            )
