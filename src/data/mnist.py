""" MNIST is a dataset of handwritten digits available on http://yann.lecun.com/exdb/mnist/. """
from __future__ import division

import os
import struct
import numpy as np

fname_train_img = "train-images-idx3-ubyte"
fname_train_lbl = "train-labels-idx1-ubyte"
fname_test_img = "t10k-images-idx3-ubyte"
fname_test_lbl = "t10k-labels-idx1-ubyte"


class mnist:
    """
    Object that contains functionality to load training and testing set of the MNIST data set,
    if present as downloaded from http://yann.lecun.com/exdb/mnist/.

    Args:
      path: String, Path to MNIST data set.
    """

    def __init__(self, path):
        self.path = os.path.expanduser(path)

    def _prepare_data(self, fname_lbl, fname_img, flatten=True, integral=True):
        # > : big endian
        # I : unsigned int

        with open(os.path.join(self.path, fname_lbl), 'rb') as fobj:
            magic, num = struct.unpack(">II", fobj.read(8))
            lbl = np.fromfile(fobj, dtype=np.uint8)

        with open(os.path.join(self.path, fname_img), 'rb') as fobj:
            magic, num, cols, rows = struct.unpack(">IIII", fobj.read(16))
            if flatten:
                img = np.fromfile(fobj, dtype=np.uint8).reshape(num, rows*cols)
            else:
                img = np.fromfile(fobj, dtype=np.uint8).reshape(num, rows, cols)
                img = np.expand_dims(img, axis=3)

        if not integral:
            img = (img / 255.0).astype(np.float16)

        return (img, lbl)

    def train(self, flatten=True, integral=True):
        """ Returns training data.

        Args:
          flatten: boolean, If True set is returned as (k, 28*28) array.
          integral: boolean, If True array is of integral type.

        Return:
          (imgs, lbls): tuple of numpy arrays, Either (60000, 28, 28, 1) or (60000, 784) and (60000,)
        """
        return self._prepare_data(fname_train_lbl, fname_train_img, flatten=flatten, integral=integral)

    def test(self, flatten=True, integral=True):
        """ Returns test data.

        Args:
          flatten: boolean, If True set is returned as (k, 28*28) array.
          integral: boolean, If True array is of integral type.

        Return:
          (imgs, lbls): tuple of numpy arrays, Either (10000, 28, 28, 1) or (10000, 784) and (10000,)
        """
        return self._prepare_data(fname_test_lbl, fname_test_img, flatten=flatten, integral=integral)

    def full(self, flatten=True, integral=True):
        trn = self._prepare_data(fname_train_lbl, fname_train_img, flatten=flatten, integral=integral)
        tst = self._prepare_data(fname_test_lbl, fname_test_img, flatten=flatten, integral=integral)
        imgs = np.vstack([trn[0], tst[0]])
        lbls = np.hstack([trn[1], tst[1]])
        return (imgs, lbls)


if __name__ == "__main__":
    mnist = mnist("~/data/mnist/")
    img, lbl = mnist.train(flatten=False)
    print(img.shape, lbl.shape)
    img, lbl = mnist.test()
    print(img.shape, lbl.shape)
