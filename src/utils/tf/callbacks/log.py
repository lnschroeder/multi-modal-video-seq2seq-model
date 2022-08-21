import os

import tensorflow as tf
from tensorflow import keras


class SetStepEpoch(keras.callbacks.Callback):
    """
    Author of class: Manuel Woellhaf
    """

    def __init__(self, epoch=0, steps_per_epoch=10, **kwargs):
        super().__init__(**kwargs)
        step = steps_per_epoch * epoch + 1

        self.global_epoch = epoch  # tf.Variable(initial_value=epoch, trainable=False, dtype=tf.int64, name="global_epoch")
        self.global_step = step  # tf.Variable(initial_value=step, trainable=False, dtype=tf.int64, name="global_step")

        self.steps_per_epoch = steps_per_epoch

        tf.summary.experimental.set_step(self.global_step)

    def on_train_begin(self, logs=None):
        tf.summary.experimental.set_step(self.global_step)

    def on_batch_end(self, batch, logs=None):
        # self.global_step = batch + self.steps_per_epoch * self.global_epoch
        self.global_step += 1  # .assign_add(1)
        tf.summary.experimental.set_step(self.global_step)

    def on_epoch_end(self, epoch, logs=None):
        self.global_epoch += 1  # .assign_add(1)


class SaveImageSeriesCallback(keras.callbacks.Callback):

    def __init__(self, batch, params, max_outputs=3, save_combined=False):
        """
        Saves the input, target, and output sequences as one image in TensorBoard.

        Args:
            batch: data that contains the sequences
            params: params dict
            max_outputs: number of samples that should be displayed in TensorBoard
            save_combined: if True, saves the inputs, target, and output in one image (stacked in that order)
        """
        super(SaveImageSeriesCallback, self).__init__()

        self.batch = next(batch.unbatch().take(max_outputs).batch(max_outputs).as_numpy_iterator())
        self.freq = params['HIST_FREQ']
        self.split_timestep = None if params['BASE_MODEL'] == 'autoencoder' else params['SEQ_LEN'] - params['PRED_LEN']
        self.max_outputs = max_outputs
        self.save_combined = save_combined
        self.simple = 'DATASET_I' not in params

        if self.simple:
            self.sensors_in = ['ae']
            self.sensor_out = 'ae'
            self.target_video = self.batch
            self.input_videos = [self.batch]
        else:
            self.sensors_in = params['DATASET_I']
            self.sensor_out = params['DATASET_O']
            self.target_video = self.batch['labels']

            self.input_videos = []
            channels_list = self.batch['sensor_channels'][0]
            start_channel = 0
            for channels in channels_list:
                input_video = self.batch['inputs'][..., start_channel:start_channel + channels]
                start_channel += channels
                self.input_videos.append(input_video)

        self.convert_grayscale_to_rgb = any([(series.shape[-1] == 3) for series in self.input_videos + [self.target_video]])
        self._file_writer = tf.summary.create_file_writer(os.path.join(params['LOG_DIR'], "tb", "imgs"))
        self._series_inputs = []
        self._series_target = None

    def on_train_begin(self, logs=None):
        with self._file_writer.as_default():
            # save input videos
            i = 1
            for sensor_in, input_video in zip(self.sensors_in, self.input_videos):
                series_input = self._get_video_as_image(input_video)
                self._series_inputs.append(series_input)
                tf.summary.image(f'{i}: input_{sensor_in}', series_input, max_outputs=self.max_outputs)
                i += 1

            # save target videos
            series_target = self._get_video_as_image(self.target_video)
            self._series_target = series_target
            tf.summary.image(f'{i}: target_{self.sensor_out}', self._series_target, max_outputs=self.max_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.freq) == 0:
            with self._file_writer.as_default():
                # save output videos
                result = self.model(self.batch, training=False)
                output_video = result['outputs'] if isinstance(result, dict) else result
                series_output = self._get_video_as_image(output_video)
                tf.summary.image(f'{len(self.sensors_in) + 2}: output_{self.sensor_out}', series_output, max_outputs=self.max_outputs)

                # save combined video
                if self.save_combined:
                    series_combined = tf.concat(self._series_inputs + [self._series_target, series_output], axis=1)
                    names_combined = '_'.join(self.sensors_in + [self.sensor_out])
                    tf.summary.image(f'{0}: {names_combined}', series_combined, max_outputs=self.max_outputs)

    def _get_video_as_image(self, video):
        b, t, h, w, c = video.shape
        video_as_image = tf.transpose(video, [0, 2, 1, 3, 4])
        video_as_image = tf.reshape(video_as_image, [b, h, w * t, c])

        if self.split_timestep:
            split_marking = tf.ones([b, h, w // 16, c])
            video_as_image = tf.concat([video_as_image[:, :, :w * self.split_timestep],
                                        split_marking,
                                        video_as_image[:, :, w * self.split_timestep:]], axis=2)

        if c == 1 and self.convert_grayscale_to_rgb:
            video_as_image = tf.image.grayscale_to_rgb(video_as_image)

        return video_as_image
