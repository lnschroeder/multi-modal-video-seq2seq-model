"""
Authors of file: Manuel Woellhaf and Lars Niklas Schroeder
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from . import kth
from . import carla


def load(params, mode, batch_size=None, as_supervised=False):
    """
    Returns the data that is specified by params

    Args:
        params: a dict with a 'DATASET' key that is a dict containing the necessary keys for the dataset
        mode: specifies the subset of the dataset 'train', 'val', 'test'

    Returns:
        returns data object and, depending on the dataset, features or labels object
    """

    if params['DATASET'] == 'whitdots':
        return load_whitdots(params, mode, batch_size=batch_size,  as_supervised=as_supervised)
    if params['DATASET'] == 'moving_mnist':
        return load_moving_mnist(params, mode, batch_size=batch_size)
    if params['DATASET'] == 'kth':
        return load_kth(params, mode, batch_size=batch_size, as_supervised=as_supervised)
    if params['DATASET'] == 'ucf':
        return load_ucf(params, mode, batch_size=batch_size, as_supervised=as_supervised)
    if 'carla' in params['DATASET']:
        return load_carla(params, mode)

    raise NotImplementedError('No dataset loader for: ' + params['DATASET'])


def load_whitdots(params, mode='train', batch_size=None, as_supervised=False):

    batch_size = params['BATCH_SIZE'] if batch_size is None else batch_size

    it, iv = int(1e4), int(1e3)

    base = params['DATA_DIR'] if 'DATA_DIR' in params else 'data'

    if params['DISTRIBUTION'] == 'deterministic':
        data = (
            np.load('data/simple_worlds/whit_dots_det_n5_l20_v2_s32_d1_data.npy'),
            np.load('data/simple_worlds/whit_dots_det_n5_l20_v2_s32_d1_labels.npy')
        )
    if params['DISTRIBUTION'] == 'binomial':
        data = (
            np.load('data/simple_worlds/whit_dots_bin_n5_l20_v2_s32_d1_data.npy'),
            np.load('data/simple_worlds/whit_dots_bin_n5_l20_v2_s32_d1_labels.npy')
        )
    if params['DISTRIBUTION'] == 'gaussian':
        data = (
            np.load('data/simple_worlds/whit_dots_gau_n5_l20_v2_s32_d1_data.npy'),
            np.load('data/simple_worlds/whit_dots_gau_n5_l20_v2_s32_d1_labels.npy')
        )
    if params['DISTRIBUTION'] == 'uniform':
        data = (
            np.load('data/simple_worlds/whit_dots_uni_n5_l20_v2_s32_d1_data.npy'),
            np.load('data/simple_worlds/whit_dots_uni_n5_l20_v2_s32_d1_labels.npy')
        )
    if params['DISTRIBUTION'] == 'test':
        it, iv = 5, 8
        data = (
            np.load(os.path.join(base, 'simple_worlds/whit_dots_det_n1_l20_v2_s32_d1_data.npy')),
            np.load(os.path.join(base, 'simple_worlds/whit_dots_det_n1_l20_v2_s32_d1_labels.npy'))
        )
    if params['DISTRIBUTION'] == 'test3':
        it, iv = 100, 10
        data = (
            np.load(os.path.join(base, 'simple_worlds/whit_dots_det_n3_l20_v3_s64_d1_data.npy')),
            np.load(os.path.join(base, 'simple_worlds/whit_dots_det_n3_l20_v3_s64_d1_labels.npy'))
        )

    if mode == 'train':
        data = {'inputs': data[0][:-it], 'labels': data[1][:-it]}
    elif mode == 'val':
        data = {'inputs': data[0][-it:-iv], 'labels': data[1][-it:-iv]}
    elif mode == 'test':
        data = {'inputs': data[0][-iv:], 'labels': data[1][-iv:]}

    if not as_supervised:
        data = data['inputs']

    dataset = tf.data.Dataset.from_tensor_slices(data)

    if mode != 'test':
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_moving_mnist(params, mode='train', batch_size=None):
    batch_size = params['BATCH_SIZE'] if batch_size is None else batch_size

    if mode == 'train':
        data = np.load('data/simple_worlds/moving_mnist_det_40k_l20_train.npz')['arr_0']
        data = np.reshape(data, [-1, params['SEQ_LEN'], data.shape[-2], data.shape[-1], 1])[:-1000]
        data = data.astype(np.float32)/255.0
        dataset = tf.data.Dataset.from_tensor_slices(data)

    if mode == 'val':
        data = np.load('data/simple_worlds/moving_mnist_det_40k_l20_train.npz')['arr_0']
        data = np.reshape(data, [-1, params['SEQ_LEN'], data.shape[-2], data.shape[-1], 1])[-1000:]
        data = data.astype(np.float32)/255.0
        dataset = tf.data.Dataset.from_tensor_slices(data)

    if mode == 'test':
        dataset, data_info = tfds.load(
            'moving_mnist', split=mode, data_dir=params['DATA_DIR'], shuffle_files=False, as_supervised=False, with_info=True
        )
        def prepare_sample(sample):
            sample = sample['video']
            # start = tf.random.uniform([], 0, tf.shape(sample)[0]-params['SEQ_LEN'], dtype=tf.int32)
            # sample = sample[start:start+params['SEQ_LEN']]
            center = tf.shape(sample)[0]//2
            sample = sample[center-params['SEQ_LEN']//2:center+params['SEQ_LEN']//2]
            sample = tf.cast(sample, tf.float32)/255.0
            return sample

        dataset = dataset.map(prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset


def load_kth(params, mode='train', batch_size=None, as_supervised=False):

    batch_size = params['BATCH_SIZE'] if batch_size is None else batch_size
    split = 'trainval' if mode == 'train' else 'test'
    split = params['SPLIT'] + '_' + split if 'SPLIT' in params else split

    centered = params['SEQ_CENTERED'] if 'SEQ_CENTERED' in params else True

    if params['DIM_OBSERVATION'][0] == 64:
        set_name = params['DATASET'] + '/small'#_no_seqsplits'
    elif params['DIM_OBSERVATION'][0] == 128:
        set_name = params['DATASET'] + '/large_no_seqsplits'
    else:
        raise NotImplementedError('KTH data set is available in 64x64 or 128x128.')

    shuffle = mode != 'test'
    dataset, data_info = tfds.load(
        set_name, split=split,
        data_dir=params['DATA_DIR'], shuffle_files=shuffle,
        as_supervised=as_supervised, with_info=True
    )

    def _process_video(video):
        sample = video[..., 0:1]
        if not centered:
            start = tf.random.uniform([], 0, tf.math.maximum(1, tf.shape(sample)[0]-params['SEQ_LEN']), dtype=tf.int32)
        elif centered == 'start':
            start = 0
        elif centered == 'end':
            start = tf.shape(sample)[0]-params['SEQ_LEN']
        else:
            center = tf.shape(sample)[0]//2
            start = center-params['SEQ_LEN']//2

        sample = sample[start:start+params['SEQ_LEN']]
        sample = tf.cast(sample, tf.float32)/255.0        
        return sample

    def _prepare_sample(sample, label=None):
        if as_supervised:
            return {'inputs': _process_video(sample), 'labels': label}
        else:
            return _process_video(sample['video'])

    dataset = dataset.map(_prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if as_supervised:
        dataset = dataset.filter(lambda x: tf.shape(x['inputs'])[0] >= params['SEQ_LEN'])
    else:
        dataset = dataset.filter(lambda x: tf.shape(x)[0] >= params['SEQ_LEN'])

    if mode != 'test':
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_ucf(params, mode='train', batch_size=None, as_supervised=False):

    batch_size = params['BATCH_SIZE'] if batch_size is None else batch_size
    mode = 'test' if mode == 'val' else mode
    shuffle = mode != 'test'
    config = tfds.download.DownloadConfig(verify_ssl=False)
    dataset, data_info = tfds.load(
        'ucf101/ucf101_1', split=mode, data_dir=params['DATA_DIR'], shuffle_files=shuffle, with_info=True,
        download_and_prepare_kwargs={"download_config": config}
    )

    centered = params['SEQ_CENTERED'] if 'SEQ_CENTERED' in params else True

    def _process_video(sample):
        if not centered:
            start = tf.random.uniform([], 0, tf.math.maximum(1, tf.shape(sample)[0]-params['SEQ_LEN']), dtype=tf.int32)
        elif centered == 'start':
            start = 0
        elif centered == 'end':
            start = tf.shape(sample)[0]-params['SEQ_LEN']
        else:
            center = tf.shape(sample)[0]//2
            start = center-params['SEQ_LEN']//2

        sample = sample[start:start+params['SEQ_LEN']]
        sample = tf.image.resize(sample, (params['DIM_OBSERVATION'][0], params['DIM_OBSERVATION'][1]))
        sample = tf.cast(sample, tf.float32)/255.0
        return sample

    def _prepare_sample(sample, label=None):
        if as_supervised:
            return {'inputs': _process_video(sample['video']), 'labels': sample['label']}
        else:
            return _process_video(sample['video'])

    dataset = dataset.map(_prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if mode != 'test':
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def load_carla(params, mode='train'):
    """
    Args:
        params: DATASET_I is a list of input sensors: rgb, dep, ofl, isg/isg_edges (isg_edges preprocesses isg)
            DATASET_O is one of the above sensors
    """
    if not params['DATASET'].endswith('default4') and mode == 'val':
        mode = 'test'

    simple = 'DATASET_I' not in params
    input_sensors = ['rgb'] if simple else params['DATASET_I']
    output_sensor = 'rgb' if simple else params['DATASET_O']
    sensors = set(input_sensors + [output_sensor])
    feature_subset = {'sample_info', 'frame_info', *sensors}
    centered = params['SEQ_CENTERED']
    shuffle = (mode != 'test')

    builder = tfds.builder(
        name=params['DATASET'],
        data_dir=params['DATA_DIR']
    )
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            extract_dir=os.path.join(params['DATA_DIR'], 'downloads', 'extracted', params['DATASET']),
            verify_ssl=False
        )
    )
    dataset = builder.as_dataset(
        split=mode,
        shuffle_files=shuffle,
        decoders=tfds.decode.PartialDecoding(dict.fromkeys(feature_subset, True))
    )
    # print(builder.info)

    def prepare_sample(sample, start=None):
        videos = {}

        if start is None:
            start = _get_subsample_start(
                sample_len=tf.shape(sample['frame_info']['frame'])[0],
                subsample_len=params['SEQ_LEN'],
                centered=centered
            )

        frame_info = {k: v[start:start+params['SEQ_LEN']] for k, v in sample['frame_info'].items()}

        for sensor in sensors:
            video = sample[sensor]
            video = video[start:start+params['SEQ_LEN']]
            if params['DIM_OBSERVATION'][:2] != [128, 128]:
                video = tf.image.resize(video, params['DIM_OBSERVATION'][:2])
                if sensor == 'isg_edges':
                    video = tf.where(video < 128, 0, 255)
            video = tf.cast(video, tf.float32) / 255.0
            videos[sensor] = video

        sensor_channels = [videos[sensor].shape[-1] for sensor in input_sensors]
        data_in = tf.concat([videos[sensor] for sensor in input_sensors], -1)
        data_out = videos[output_sensor]

        if simple:
            return data_in
        else:
            return {
                'inputs': data_in,
                'labels': data_out,
                'sensor_channels': sensor_channels,
                'sample_info': sample['sample_info'],
                'frame_info': frame_info
            }

    # random subsampling needs to be done sequentially to be deterministic
    # see: https://github.com/tensorflow/tensorflow/issues/13932
    if not centered:
        dataset = dataset.map(
            map_func=lambda sample: (
                sample,
                _get_subsample_start(
                    sample_len=tf.shape(sample['frame_info']['frame'])[0],
                    subsample_len=params['SEQ_LEN'],
                    centered=centered)
            ),
            num_parallel_calls=1
        )

    dataset = dataset.map(
        map_func=prepare_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if mode != 'test':
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(params['BATCH_SIZE'])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if mode != 'test':
        dataset = dataset.repeat()

    return dataset


def _get_subsample_start(sample_len, subsample_len, centered):
    if centered == True:
        center = sample_len // 2
        start = center - subsample_len // 2
    elif centered == False:
        start = tf.random.uniform([], 0, tf.math.maximum(1, sample_len - subsample_len), dtype=tf.int32)
    elif centered == 'start':
        start = 0
    elif centered == 'end':
        start = sample_len - subsample_len
    else:
        raise NotImplementedError(f'Centered option `{centered}` is not implemented')
    return start


if __name__ == '__main__':
    params = {}
    params['SEQ_LEN'] = 20
    params['BATCH_SIZE'] = 3
    params['DATA_DIR'] = './'
    params['DATASET'] = 'whitdots'
    params['DISTRIBUTION'] = 'deterministic'
    load(params, mode='test')
