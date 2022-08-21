import os
import h5py
from tqdm import tqdm

import tensorflow as tf


def _batch(inputs, labels=None, batch_size=1):
    """
    Author of def: Manuel Woellhaf
    """

    for i in range(0, inputs.shape[0], batch_size):
        s, e = i*batch_size, i*batch_size + batch_size

        if labels is None:
            yield inputs[s:e]
        else:
            yield {'inputs': inputs[s:e], 'labels': labels[s:e]}


def write_evaluation_output(out_dir, model, inputs, context=None, batch_size=1, name='eval_data'):
    """
    Author of def: Manuel Woellhaf but modified by Lars Niklas Schroeder

    Args:
      out_dir: String
      model: tensorflow model
      inputs: numpy array or tensorflow dataset
      context: additional data to be saved
      batch_size: Integer (number of samples needs to be a multiple of batch_size)
        Only used if inputs is not a tensorflow dataset.
      name: String file name
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with h5py.File(os.path.join(out_dir, name+'.h5'), 'w') as hf:

        if context is not None:
            hf.create_dataset('context', data=context)

        # get iterator
        if isinstance(inputs, tf.data.Dataset):
            itr = inputs.as_numpy_iterator()
        elif isinstance(inputs, dict):
            itr = _batch(inputs['inputs'], inputs['labels'], batch_size=batch_size)
        else:
            itr = _batch(inputs, labels=None, batch_size=batch_size)

        # evaluate one sample to determine data structure
        batch = next(itr)
        is_labeled = isinstance(batch, dict)
        result = model(batch, training=False)
        batch_size = batch['inputs'].shape[0] if is_labeled else batch.shape[0]

        for key in batch:
            if isinstance(batch[key], dict):
                for meta_key in batch[key]:
                    hf.create_dataset(meta_key, data=batch[key][meta_key], maxshape=(None,)+batch[key][meta_key].shape[1:])
            else:
                hf.create_dataset(key, data=batch[key], maxshape=(None,)+batch[key].shape[1:])

        if isinstance(result, dict):
            for key in result:
                hf.create_dataset(key, data=result[key], maxshape=(None,)+result[key].shape[1:])
        else:
            hf.create_dataset('outputs', data=result, maxshape=(None,)+result.shape[1:])

        # loop over the other samples
        for i, batch in tqdm(enumerate(itr, start=1)):
            index = i*batch_size
            result = model(batch, training=False)
            b_size = min(batch_size, batch['inputs'].shape[0])

            for key in batch.keys():
                if isinstance(batch[key], dict):
                    for meta_key in batch[key]:
                        hf[meta_key].resize(hf[meta_key].shape[0] + b_size, axis=0)
                        hf[meta_key][index:index + b_size] = batch[key][meta_key] if is_labeled else batch
                else:
                    hf[key].resize(hf[key].shape[0] + b_size, axis=0)
                    hf[key][index:index + b_size] = batch[key] if is_labeled else batch

            if isinstance(result, dict):
                for key in result:
                    hf[key].resize(hf[key].shape[0]+b_size, axis=0)
                    hf[key][index:index+b_size] = result[key]
            else:
                hf['outputs'].resize(hf['outputs'].shape[0]+b_size, axis=0)
                hf['outputs'][index:index+b_size] = result
