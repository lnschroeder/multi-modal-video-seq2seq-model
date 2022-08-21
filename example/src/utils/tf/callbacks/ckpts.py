"""
Author of file: Manuel Woellhaf but modified by Lars Niklas Schroeder
"""
import os
import glob

import logging
import tensorflow as tf

from tensorflow import keras


class TensorflowCheckpoint(keras.callbacks.Callback):
    """
    This functionality should be handled by built in TF classes, however for some reason those methods dont
    work or do not save the optimizer state for some of my models. Therefor the replicate here.

    Monitor works only on epoch level to avoid an insane amount of garbage collection.

    Args:
      save_freq : Integer
      monitor   : String metric (default None)
      mode      : String 'max' or 'min'
    """

    def __init__(self, filepath, last_epoch, save_freq=1, monitor=None, mode='min', expect_partial=False):
        super().__init__()

        self.filepath = filepath
        self.last_epoch = last_epoch
        self.save_freq = save_freq
        self.monitor = monitor
        self.mode = mode
        self.expect_partial = expect_partial

        self._monitor_val_accu = []
        self._monitor_val_curr = None
        self._monitor_val_best = None
        self._monitor_ckpt_best = None

    def _save_checkpoint(self, num, prefix='ckpt', logs=None):
        return self.ckpt.save(os.path.join(self.filepath, prefix+'.{0}.'.format(num)))

    def _garbage_collector(self):
        ckpt_list = sorted([f for f in os.listdir(self.filepath) if 'ckpt.' in f and 'index' in f], key=lambda x: int(x.split('.')[1]))[:-1]

        if self.save_freq > 0:
            ckpt_list = [f for f in ckpt_list if (int(f.split('.')[1]) % self.save_freq) != 0]

        if self._monitor_ckpt_best:
            ckpt_list = [f for f in ckpt_list if f.split('.index')[0] != os.path.basename(self._monitor_ckpt_best)]

        for ckpt in ckpt_list:
            file_list = glob.glob(os.path.join(self.filepath, ckpt.split('.index')[0]+'*'))
            for f in file_list:
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass

    def _check_monitor(self):

        if self._monitor_val_best is None:
            save_ckpt = True
        elif self.mode == 'max':
            save_ckpt = self._monitor_val_best < self._monitor_val_curr
        elif self.mode == 'min':
            save_ckpt = self._monitor_val_best > self._monitor_val_curr
        else:
            save_ckpt = False
            logging.warning(f'`{self.mode}` mode for checkpointing not recognized.')

        if save_ckpt:
            self._monitor_val_best = self._monitor_val_curr

        return save_ckpt

    def set_model(self, model):
        self.model = model
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, model=model)

    def on_train_begin(self, logs=None):
        latest_ckpt = tf.train.latest_checkpoint(self.filepath)

        if latest_ckpt:
            status = self.ckpt.restore(latest_ckpt)
            if self.expect_partial:
                status.expect_partial()
            else:
                status.assert_consumed()
        else:
            self._save_checkpoint(0)

    def on_test_batch_end(self, batch, logs=None):
        if self.monitor in logs.keys():
            self._monitor_val_accu.append(logs[self.monitor])
        else:
            logging.warning(f'{self.monitor} cannot be found in logs.')

    def on_test_end(self, logs=None):
        if self._monitor_val_accu:
            self._monitor_val_curr = sum(self._monitor_val_accu)/len(self._monitor_val_accu)
            self._monitor_val_accu = []

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor and self._check_monitor():
            self._monitor_ckpt_best = self._save_checkpoint(epoch+1)
            self._garbage_collector()
        elif self.save_freq > 0 and (epoch+1 == self.last_epoch or ((epoch+1) % self.save_freq) == 0):
            self._save_checkpoint(epoch+1)
