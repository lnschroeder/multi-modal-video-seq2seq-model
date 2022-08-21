import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Info (1), Warning (2), Error (3) messages

import numpy as np
import tensorflow as tf
import random
import yaml

from tensorflow import keras

from ..data.loader import load
from ..utils.tf.callbacks import TensorflowCheckpoint
from ..utils.tf.callbacks.log import SetStepEpoch, SaveImageSeriesCallback
from ..utils.tf.lr_schedules import ConstLearningRate
from ..utils.tf.ckpts import get_ckpt, get_latest_ckpt, get_best_ckpt
from ..utils.plot import plot_results

from .spatio_temporal_models import SrivastavaComposite, SrivastavaAE
from .utils.fileio import write_evaluation_output


def get_compiled_model(params):
    """
    Returns compiled model.
    Args:
        params: a dict containing all parameters for the model (params.yml+config.yml for YASSMLTK)

    Returns:
        the compiled model
    """
    if params['BASE_MODEL'] == 'autoencoder':
        base_model = SrivastavaAE
    elif params['BASE_MODEL'] == 'composite':
        base_model = SrivastavaComposite
    else:
        raise NotImplementedError

    model = base_model(params, log_dir=params['LOG_DIR'])

    if isinstance(params['LR'], float):
        lr = ConstLearningRate(params)
    elif isinstance(params['LR'], list):
        lr = keras.optimizers.schedules.PiecewiseConstantDecay(params['LR_BOUNDARIES'], params['LR'])
    else:
        raise NotImplementedError

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=None, metrics=None, run_eagerly=True)  # TODO run eagerly

    return model


def print_start(params, mode):
    seed = random.randint(0, 1000000)
    open(os.path.join(params['LOG_DIR'], str(seed)), 'w').close()
    print(f'--- {mode} / {str(seed)} ---')
    print(yaml.dump(params, default_flow_style=False))
    print('--- ---------------- ---')


def evaluate(params):
    """
    This method gets called second (and last) from YASSMLTK
    It starts evaluation.

    Args:
        params: a dict containing all parameters for the model (params.yml+config.yml for YASSMLTK)
    """
    print_start(params, 'evaluate')
    tf.random.set_seed(params['RANDOM_SEED'])
    np.random.seed(params['RANDOM_SEED'])

    # load data
    data_test = load(params, mode='test')

    # laod model
    model = get_compiled_model(params)

    # load checkpoint
    eval_ckpt = params['EVAL_CKPT'] if 'EVAL_CKPT' in params else 'best'
    if eval_ckpt == 'best':
        log_file_path = os.path.join(params['LOG_DIR'], 'log.csv')
        metric = params['EVAL_METRIC'] if 'EVAL_METRIC' in params else 'val_loss'
        mode = params['EVAL_MODE'] if 'EVAL_MODE' in params else 'min'
        ckpt_nr, ckpt_name = get_best_ckpt(params['CKPT_DIR'], log_file_path, metric=metric, mode=mode)
    elif eval_ckpt == 'latest':
        ckpt_nr, ckpt_name = get_latest_ckpt(params['CKPT_DIR'])
    elif isinstance(eval_ckpt, int):
        ckpt_nr, ckpt_name = get_ckpt(params['CKPT_DIR'], eval_ckpt)
    else:
        raise NotImplementedError(f'Loading checkpoint {eval_ckpt} is not possible')

    ckpt_path = os.path.join(params['CKPT_DIR'], ckpt_name)
    print(f'Using {ckpt_path} for evaluation')

    # initialize model
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, model=model)
    ckpt.restore(ckpt_path).expect_partial()

    # evaluate
    write_evaluation_output(params['EVAL_DIR'], model, data_test, name='eval_data_' + str(ckpt_nr))
    plot_results(params, epoch=ckpt_nr)
    print('- Finished evaluation - ')


def train(params):
    """
    This method gets called first from YASSMLTK
    It starts training.

    Args:
        params: a dict containing all parameters for the model (params.yml+config.yml for YASSMLTK)

    """
    print_start(params, 'train')
    tf.random.set_seed(params['RANDOM_SEED'])
    np.random.seed(params['RANDOM_SEED'])

    # create model
    model = get_compiled_model(params)

    # load data
    data_trn = load(params, mode='train')
    data_val = load(params, mode='val')

    # load checkpoint
    if not os.path.exists(params['CKPT_DIR']):
        os.makedirs(params['CKPT_DIR'])

    ckpt_nr, ckpt_name = get_latest_ckpt(params['CKPT_DIR'])
    if ckpt_nr >= 0:
        # if we use TensorflowCheckpoint we need to initialize the model
        # including the optimizer so that the callback can resume training
        model.train_on_batch(next(iter(data_trn)))

    # fit model
    verbose = 2 if os.path.exists(os.path.join(params['EXPERIMENT_DIR'], 'config.slurm')) else 1
    initial_epoch = max(0, ckpt_nr)

    val_epochs = list(range(1, params['EPOCHS'] + 1, params['VAL_FREQ']))
    if val_epochs[-1] != params['EPOCHS']:
        val_epochs.append(params['EPOCHS'])

    callbacks = _get_callbacks(
        params=params,
        initial_epoch=initial_epoch,
        sis_batch=data_val.take(1)
    )

    model.fit(
        data_trn,
        epochs=params['EPOCHS'],
        initial_epoch=initial_epoch,
        steps_per_epoch=params['STEPS_PER_EPOCH'],  # TODO
        validation_data=data_val,
        validation_freq=val_epochs,
        validation_steps=params['VAL_STEPS'],
        verbose=verbose,
        callbacks=callbacks,
    )

    print('- Finished training - ')


def _get_callbacks(params, initial_epoch, sis_batch):
    callbacks = []

    callbacks.append(tf.keras.callbacks.TensorBoard(
        os.path.join(params['LOG_DIR'], 'tb'),
        histogram_freq=params['HIST_FREQ'],
        write_graph=False,
        write_images=False,
        update_freq='batch',
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None
    ))

    callbacks.append(tf.keras.callbacks.CSVLogger(
        os.path.join(params['LOG_DIR'], 'log.csv'),
        separator=',',
        append=True
    ))

    callbacks.append(SaveImageSeriesCallback(
        batch=sis_batch,
        params=params,
        max_outputs=3,
        save_combined=True
    ))

    callbacks.append(TensorflowCheckpoint(
        params['CKPT_DIR'],
        last_epoch=params['EPOCHS'],
        save_freq=params['CKPT_FREQ'],
        monitor='loss',
        mode='min',
        expect_partial=params['EXPECT_PARTIAL'] if 'EXPECT_PARTIAL' in params else False
    ))

    callbacks.append(SetStepEpoch(
        epoch=initial_epoch,
        steps_per_epoch=params['STEPS_PER_EPOCH']
    ))

    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    # EarlyStopping with patience < 1 always stop after first epoch
    if 'PATIENCE' in params and params['PATIENCE'] > 0:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params['PATIENCE'],
            verbose=1,
        ))

    return callbacks
