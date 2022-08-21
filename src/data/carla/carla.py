"""carla dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import logging
import yaml
import csv
import os

from pathlib import Path

_DESCRIPTION = """
A CARLA dataset by Lars Niklas Schroder.
Consists of sequences of images with different visual modalities and ego motion data of all vehicles.
"""

_CITATION = """
"""

# See Carla documentation for updates:
# https://github.com/carla-simulator/carla/blob/master/LibCarla/source/carla/image/CityScapesPalette.h
CITYSCAPES_PALETTE_MAP = {
    0:  [  0,   0,   0],  # unlabeled     =   0,
    1:  [ 70,  70,  70],  # building      =   1,
    2:  [100,  40,  40],  # fence         =   2,
    3:  [ 55,  90,  80],  # other         =   3,
    4:  [220,  20,  60],  # pedestrian    =   4,
    5:  [153, 153, 153],  # pole          =   5,
    6:  [157, 234,  50],  # road line     =   6,
    7:  [128,  64, 128],  # road          =   7,
    8:  [244,  35, 232],  # sidewalk      =   8,
    9:  [107, 142,  35],  # vegetation    =   9,
    10: [  0,   0, 142],  # vehicle       =  10,
    11: [102, 102, 156],  # wall          =  11,
    12: [220, 220,   0],  # traffic sign  =  12,
    13: [ 70, 130, 180],  # sky           =  13,
    14: [ 81,   0,  81],  # ground        =  14,
    15: [150, 100, 100],  # bridge        =  15,
    16: [230, 150, 140],  # rail track    =  16,
    17: [180, 165, 180],  # guard rail    =  17,
    18: [250, 170,  30],  # traffic light =  18,
    19: [110, 190, 160],  # static        =  19,
    20: [170, 120,  50],  # dynamic       =  20,
    21: [ 45,  60, 150],  # water         =  21,
    22: [145, 170, 100],  # terrain       =  22,
    # [  0,   0,  70], # truck
    # [  0,   0,  90], # caravan
    # [  0,   0, 110], # trailer
    # [  0,   0, 142], # license plate
    # [  0,   0, 230], # motorcycle
    # [  0,  60, 100], # bus
    # [  0,  80, 100], # train
    # [119,  11,  32], # bicycle
    # [150, 120,  90], # tunnel
    # [153, 153, 153], # polegroup
    # [250, 170, 160], # parking
    # [255,   0,   0], # rider
}

SCP_RGB = [
    tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=[*CITYSCAPES_PALETTE_MAP],
            values=[v[i] for v in CITYSCAPES_PALETTE_MAP.values()]
        ),
        default_value=tf.constant(0),
    ) for i in range(3)
]


def get_features_dict(video_shape, length, include_actors=False):
    features_dict = {
        # These are the features of your dataset like images, labels ...
        'rgb': tfds.features.Video(
            (*video_shape, 3),
            encoding_format='png'
        ),

        'dep': tfds.features.Video(
            (*video_shape, 1),
            encoding_format='png'
        ),

        'isg': tfds.features.Video(
            (*video_shape, 3),
            encoding_format='png'
        ),

        'isg_edges': tfds.features.Video(
            (*video_shape, 1),
            encoding_format='png'
        ),

        'ssg': tfds.features.Video(
            (*video_shape, 3),
            encoding_format='png'
        ),

        'ofl': tfds.features.Video(
            (*video_shape, 3),
            encoding_format='png'
        ),

        'frame_info': tfds.features.Sequence(
            feature={
                'frame': tf.int32,
                'traffic_light': tfds.features.ClassLabel(
                    names=['Red', 'Yellow', 'Green', 'Off', 'Unknown', 'None']
                ),
                'speed_limit': tf.float32,
                'speed': tf.float32
            },
            length=length
        ),

        'sample_info': tfds.features.FeaturesDict({
            '_actor_id': tf.int32,
            '_hash': tf.string,
            '_n_vehicles_actual': tf.int32,
            '_n_walkers_actual': tf.int32,
            'cam_pitch': tf.float32,
            'cam_yaw': tf.float32,
            'cam_roll': tf.float32,
            'cam_x': tf.float32,
            'cam_y': tf.float32,
            'cam_z': tf.float32,
            'duration': tf.int32,
            'fps': tf.int32,
            'img_h': tf.float32,
            'img_w': tf.float32,
            'fov': tf.float32,
            'map_name': tf.string,
            'n_vehicles': tf.int32,
            'n_walkers': tf.int32,
            'seed': tf.int32,
            'weather': tf.string,
            'speed_diff': tf.float32,
            'split': tf.string
        })
    }

    if include_actors:
        features_dict['actors'] = tfds.features.Sequence(tfds.features.Sequence(
            feature={
                'id': tf.int32,
                'type_id': tf.string,
                # 'attribs': tf.string,
                # 'fwd_x': tf.float32,
                # 'fwd_y': tf.float32,
                # 'fwd_z': tf.float32,
                # 'rgt_x': tf.float32,
                # 'rgt_y': tf.float32,
                # 'rgt_z': tf.float32,
                # 'upp_x': tf.float32,
                # 'upp_y': tf.float32,
                # 'upp_z': tf.float32,
                # 'rot_p': tf.float32,
                # 'rot_y': tf.float32,
                # 'rot_r': tf.float32,
                'loc_x': tf.float32,
                'loc_y': tf.float32,
                'loc_z': tf.float32,
                # 'vel_x': tf.float32,
                # 'vel_y': tf.float32,
                # 'vel_z': tf.float32,
                # 'acc_x': tf.float32,
                # 'acc_y': tf.float32,
                # 'acc_z': tf.float32,
                # 'agv_x': tf.float32,
                # 'agv_y': tf.float32,
                # 'agv_z': tf.float32
                'speed': tf.float32
            }
        ))
    return features_dict


class CarlaConfig(tfds.core.BuilderConfig):
    """"Configuration for KTH video rescaling."""

    VERSION = tfds.core.Version('1.1.2')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        '1.0.1': 'subsample len to samples amount',
        '1.1.1': 'add ssg',
        '1.1.2': 'switch to manual download',
    }

    def __init__(self, download_urls, manual_download=False, width=None, height=None, sub_sample_len=None, reduce_fps_by=1, **kwargs):
        """The parameters specifying how the dataset will be processed.
        Args:
          width: An integer with the width or None.
          height: An integer with the height or None.
          sub_sample_len: An integer with length
          **kwargs: Passed on to the constructor of `BuilderConfig`.
        """

        super(CarlaConfig, self).__init__(
            version=self.VERSION,
            release_notes=self.RELEASE_NOTES,
            **kwargs
        )

        self.download_urls = download_urls
        self.manual_download = manual_download
        self.width = width or height
        self.height = height or width
        self.sub_sample_len = sub_sample_len
        self.reduce_fps_by = reduce_fps_by
        self.include_actors = self.name.endswith('_full')
        self.basename = self.name[:-len('_full')] if self.include_actors else self.name


class Carla(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for carla dataset."""

    BUILDER_CONFIGS = [
        CarlaConfig(
            name='default3',
            description='Carla dataset 128x128x300s*24fps 92 (100-8) samples (https://tubcloud.tu-berlin.de/s/EAT6J6YcJcbK3gG)',
            download_urls={
                'params.zip': 'https://tubcloud.tu-berlin.de/s/BMp8ZmZi3S3mxbq/download',
                'Town01_Opt.zip': 'https://tubcloud.tu-berlin.de/s/mzGJB8wZRDCYTwa/download',
                'Town02_Opt.zip': 'https://tubcloud.tu-berlin.de/s/ei6QxTPZ23cb9ar/download',
                'Town03_Opt.zip': 'https://tubcloud.tu-berlin.de/s/BHW7yXsHfG6E4Hs/download',
                'Town04_Opt.zip': 'https://tubcloud.tu-berlin.de/s/sZ7KkfBRYZCe6ir/download',
                'Town05_Opt.zip': 'https://tubcloud.tu-berlin.de/s/tqKXXKLDEMoPgWC/download',
                'Town06_Opt.zip': 'https://tubcloud.tu-berlin.de/s/3esEmBaFxrxzcjD/download',
                'Town07_Opt.zip': 'https://tubcloud.tu-berlin.de/s/GikmkkQXiy9AQ5T/download',
                'Town10HD_Opt.zip': 'https://tubcloud.tu-berlin.de/s/J73sPnacQKFgttt/download',
            },
            manual_download=True,
            sub_sample_len=80,
            reduce_fps_by=1,
            width=128,
            height=128
        ),

        CarlaConfig(
            name='default4',
            description='Carla dataset 128x128x300s*25fps 223 (210/7/7 - 1/0/0 (train/test/val)) samples (https://tubcloud.tu-berlin.de/s/5ygs7dbMjDJWtqX). Please generate dataset manually.',
            download_urls={
                'params.zip': 'https://tubcloud.tu-berlin.de/s/bo2o6XzBeL3Fin4/download',
                'Town01_Opt.zip': 'TODO/download',
                'Town02_Opt.zip': 'TODO/download',
                'Town03_Opt.zip': 'TODO/download',
                'Town04_Opt.zip': 'TODO/download',
                'Town05_Opt.zip': 'TODO/download',
                'Town06_Opt.zip': 'TODO/download',
                'Town07_Opt.zip': 'TODO/download',
            },
            manual_download=True,
            sub_sample_len=80,
            reduce_fps_by=1,
            width=128,
            height=128
        ),

        CarlaConfig(
            name='default3_small',
            description='Carla dataset 128x128x300s*24fps 92 (100-8) samples (https://tubcloud.tu-berlin.de/s/EAT6J6YcJcbK3gG)',
            download_urls={
                'params.zip': 'https://tubcloud.tu-berlin.de/s/BMp8ZmZi3S3mxbq/download',
                'Town01_Opt.zip': 'https://tubcloud.tu-berlin.de/s/mzGJB8wZRDCYTwa/download',
                'Town10HD_Opt.zip': 'https://tubcloud.tu-berlin.de/s/J73sPnacQKFgttt/download',
            },
            manual_download=True,
            sub_sample_len=80,
            reduce_fps_by=1,
            width=128,
            height=128
        ),

        CarlaConfig(
            name='params_dummy',
            description='Carla dataset 128x128x30s*24fps from 300s sample.',
            download_urls='https://tubcloud.tu-berlin.de/s/zr6PyeArLekyCa5/download',
            sub_sample_len=20,
            reduce_fps_by=1,
            width=128,
            height=128,
        ),

        CarlaConfig(
            name='params_dummy_small',
            description='Carla dataset 128x128x30s*24fps from 300s sample.',
            download_urls='https://tubcloud.tu-berlin.de/s/BFXQSbgjBsCTPMY/download',
            sub_sample_len=20,
            reduce_fps_by=1,
            width=128,
            height=128
        ),

        CarlaConfig(
            name='params',
            description='Carla 128x128x30s*10fps.',
            download_urls='https://tubcloud.tu-berlin.de/s/LoTtR9egBL4yWjE/download',
            sub_sample_len=30 * 10,
            reduce_fps_by=1,
            width=128,
            height=128
        ),

        CarlaConfig(
            name='params_full',  # if name ends with '_full' actor info will also be saved
            description='Carla 128x128x30s*10fps.',
            download_urls='https://tubcloud.tu-berlin.de/s/LoTtR9egBL4yWjE/download',
            sub_sample_len=30 * 10,
            reduce_fps_by=1,
            width=128,
            height=128
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        video_shape = (
            self.builder_config.sub_sample_len,
            self.builder_config.height,
            self.builder_config.width,
        )

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(get_features_dict(video_shape,
                                                                  self.builder_config.sub_sample_len,
                                                                  self.builder_config.include_actors))
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        if self.builder_config.manual_download:
            dir_path = os.path.join(dl_manager.download_dir, 'manual', 'carla', self.builder_config.name)
            paths = {key.split('.')[0]: os.path.join(dir_path, key) for key in self.builder_config.download_urls}
            paths = dl_manager.extract(paths)
        else:
            paths = dl_manager.download_and_extract(
                self.builder_config.download_urls
            )

        split_paths = {
            'train': self._generate_examples(paths, 'train'),
            'test': self._generate_examples(paths, 'test'),
        }

        if self.builder_config.name == 'default4':
            split_paths['val'] = self._generate_examples(paths, 'val')

        return split_paths

    def _generate_examples(self, paths, split):
        """Yields examples."""
        if isinstance(self.builder_config.download_urls, dict):
            sample_paths = []
            with open(os.path.join(paths['params'], 'params.csv')) as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['split'] == split:
                        town = row['map']
                        if town in paths:
                            sample_path = os.path.join(paths[town], town, row['hash'])
                            sample_paths.append(sample_path)
        else:
            split_path = os.path.join(paths, self.builder_config.basename, split)
            sample_paths = [str(sample_path) for sample_path in Path(split_path).glob('*/*')]

        for sample_path in sample_paths:
            if not os.path.isdir(sample_path):
                logging.warning(f'Sample path not found in dataset: {sample_path}')
                continue

            with open(os.path.join(sample_path, 'sample_info.yml')) as file:
                sample_info = yaml.safe_load(file)
            del sample_info['time']

            frame_info = []
            speeds = []
            with open(os.path.join(sample_path, 'frame_info.csv')) as file:
                reader = csv.DictReader(file)
                for row in reader:
                    frame_info.append({
                        'frame': int(row['frame']),
                        'traffic_light': row['traffic_light'],
                        'speed_limit': float(row['speed_limit']),
                        'speed': float(row['speed'])
                    })
                    speeds.append(float(row['speed']))

            min_frame = int(frame_info[0]['frame'])
            max_frame = int(frame_info[-1]['frame'])
            filenames = [f"{i:08d}" for i in range(min_frame, max_frame+1)]
            sample_len = len(filenames)
            sub_sample_len = self.builder_config.sub_sample_len

            actors = []
            if self.builder_config.include_actors:
                for f in filenames:
                    start_frame = []
                    with open(os.path.join(sample_path, 'actors', f + '.csv')) as file:
                        # actors.append(list(csv.DictReader(file)))
                        for row in csv.DictReader(file):
                            speed = 3.6 * tf.norm(tf.constant([float(row['vel_x']), float(row['vel_y']), float(row['vel_z'])]))
                            start_frame.append({
                                'id': row['id'],
                                'type_id': row['type_id'],
                                'loc_x': float(row['loc_x']),
                                'loc_y': float(row['loc_y']),
                                'loc_z': float(row['loc_z']),
                                'speed': float(speed)
                            })
                    actors.append(start_frame)

            # create isg_edges from isg
            paddings = tf.constant([[0, 1], [0, 1], [0, 0]])  # l/r-paddings [h, w, c]
            isg_path = os.path.join(sample_path, 'isg')

            os.makedirs(os.path.join(sample_path, 'isg_edges'), exist_ok=True)
            os.makedirs(os.path.join(sample_path, 'ssg'), exist_ok=True)

            def preprocess_isg(img_names):
                for img_name in img_names:
                    img_name += '.png'
                    isg_edges_path = os.path.join(sample_path, 'isg_edges', img_name)
                    ssg_path = os.path.join(sample_path, 'ssg', img_name)

                    force_recompute = False  # only for debugging when recomputing edges and ssg is necessary
                    isg_edges_exists = os.path.isfile(isg_edges_path) if not force_recompute else False
                    ssg_exists = os.path.isfile(ssg_path) if not force_recompute else False

                    if isg_edges_exists and ssg_exists:
                        continue

                    isg = tf.keras.preprocessing.image.load_img(os.path.join(isg_path, img_name))
                    isg = tf.keras.preprocessing.image.img_to_array(isg)

                    if not isg_edges_exists:
                        isg_padding = tf.pad(isg, paddings, 'SYMMETRIC')
                        isg_edges_h = tf.math.not_equal(isg_padding[:-1, :], isg_padding[1:, :])[:, :-1]
                        isg_edges_v = tf.math.not_equal(isg_padding[:, :-1], isg_padding[:, 1:])[:-1, :]
                        isg_edges = tf.math.logical_or(isg_edges_h, isg_edges_v)
                        isg_edges = tf.reduce_any(isg_edges, axis=-1, keepdims=True)
                        isg_edges = tf.cast(isg_edges, tf.uint8) * 255
                        tf.keras.preprocessing.image.save_img(isg_edges_path, isg_edges, scale=False)

                    if not ssg_exists:
                        csp_id = tf.convert_to_tensor(isg[..., 0], dtype=tf.int32)
                        ssg = tf.stack([ssg_c.lookup(csp_id) for ssg_c in SCP_RGB], axis=2)
                        tf.keras.preprocessing.image.save_img(ssg_path, ssg, scale=False)

            for start_frame in range(0, sample_len - sub_sample_len * self.builder_config.reduce_fps_by + 1, sub_sample_len * self.builder_config.reduce_fps_by):
                s = start_frame
                e = start_frame + sub_sample_len * self.builder_config.reduce_fps_by
                sample_key = f'{sample_path.split("/")[-1]}_{s+min_frame:08d}-{e+min_frame:08d}'

                # remove sample if car is not moving much
                if tf.math.reduce_mean(speeds[s:e]) > 0.1:
                    preprocess_isg(filenames[s:e:self.builder_config.reduce_fps_by])

                    sample = {
                        'rgb': [os.path.join(sample_path, 'rgb', f + '.png') for f in filenames[s:e:self.builder_config.reduce_fps_by]],
                        'dep': [os.path.join(sample_path, 'dep', f + '.png') for f in filenames[s:e:self.builder_config.reduce_fps_by]],
                        'ofl': [os.path.join(sample_path, 'ofl', f + '.png') for f in filenames[s:e:self.builder_config.reduce_fps_by]],
                        'isg': [os.path.join(sample_path, 'isg', f + '.png') for f in filenames[s:e:self.builder_config.reduce_fps_by]],
                        'isg_edges': [os.path.join(sample_path, 'isg_edges', f + '.png') for f in filenames[s:e:self.builder_config.reduce_fps_by]],
                        'ssg': [os.path.join(sample_path, 'ssg', f + '.png') for f in filenames[s:e:self.builder_config.reduce_fps_by]],
                        'frame_info': frame_info[s:e:self.builder_config.reduce_fps_by],
                        'sample_info': sample_info
                    }

                    if self.builder_config.include_actors:
                        sample['actors'] = actors[s:e:self.builder_config.reduce_fps_by]

                    yield sample_key, sample
