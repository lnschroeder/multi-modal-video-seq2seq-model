import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable Info (1), Warning (2), Error (3) messages
import argparse
import h5py
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import pandas as pd


def _get_input_videos_as_images(input_videos, split_timestep, sensors_in, channels_list):
    series_inputs = []
    start_channel = 0
    for sensor_in, channels in zip(sensors_in, channels_list):
        input_video = input_videos[..., start_channel:start_channel + channels]
        start_channel += channels
        series_input = _get_video_as_image(input_video, split_timestep)
        series_inputs.append(series_input)
    return series_inputs


def _get_video_as_image(video, split_timestep, convert_grayscale_to_rgb=True):
    t, h, w, c = video.shape
    video_as_image = tf.transpose(video, [1, 0, 2, 3])
    video_as_image = tf.reshape(video_as_image, [h, w * t, c])

    if split_timestep:
        split_marking = tf.ones([h, w // 16, c])
        video_as_image = tf.concat([video_as_image[:, :w * split_timestep],
                                    split_marking,
                                    video_as_image[:, w * split_timestep:]], axis=1)

    if c == 1 and convert_grayscale_to_rgb:
        video_as_image = tf.image.grayscale_to_rgb(video_as_image)

    return video_as_image


def plot_results_model_dir(d, epoch=None, amount=10):
    with open(os.path.join(d, 'params.yml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    plot_results(params, eval_dir=os.path.join(d, 'eval'), epoch=epoch, amount=amount)


def plot_results(params, eval_dir=None, epoch=None, amount=10):
    """
    Creates some evaluation results as PNGs, SVGs. Also saves Metrics

    Args:
        params: params dict
        eval_dir: directory in which the files should be written to
        epoch: the epoch number of the h5 file, which was created by write_evaluation_output (fileio.py)
        amount: amount of samples
    """
    split_timestep = None if params['BASE_MODEL'] == 'autoencoder' else params['SEQ_LEN'] - params['PRED_LEN']
    loss_name = params['LOSS'] if 'LOSS' in params else 'MSE'
    use_bce = params['USE_BCE'] if 'USE_BCE' in params else False
    amount = params['EVAL_AMOUNT'] if 'EVAL_AMOUNT' in params else amount
    sensors_in = params['DATASET_I'] if 'DATASET_I' in params else ['rgb']
    eval_dir = params['EVAL_DIR'] if eval_dir is None else eval_dir

    if epoch is None:
        eval_fns = [fn for fn in os.listdir(eval_dir) if fn.endswith('.h5')]
        eval_fns.sort(key=lambda f: int(f[:-3].split('_')[-1]))
        eval_fn = eval_fns[-1]
    else:
        eval_fn = f'eval_data_{epoch}.h5'
    eval_path = os.path.join(eval_dir, eval_fn)
    if not os.path.exists(eval_path):
        print(f"File does not exit: {eval_path}")
        raise ValueError

    if sensors_in is None:
        sensors_in = ['rgb']

    with h5py.File(eval_path, 'r') as hf:
        inputs = hf['inputs']
        targets = hf['labels']
        outputs = hf['outputs']
        speed = hf['speed']
        channels_list = hf['sensor_channels'][0]

        ssims = tf.image.ssim(targets, outputs, 1.0)
        ssim = tf.math.reduce_mean(ssims)

        psnrs = tf.image.psnr(targets, outputs, 1.0)
        psnr = tf.math.reduce_mean(psnrs)

        mse = tf.keras.losses.MeanSquaredError()(targets, outputs)

        metrics = {'mse': float(mse.numpy()), 'psnr': float(psnr.numpy()), 'ssim': float(ssim.numpy())}

        if use_bce or loss_name == 'BCE':
            bce = tf.keras.losses.BinaryCrossentropy()(targets, outputs)
            metrics['bce'] = float(bce.numpy())

        if use_bce or loss_name == 'WBCE':
            edge_weight = tf.reduce_sum(targets) / tf.size(targets, out_type=tf.float32)
            weights = tf.where(tf.equal(targets, 1.), 1 - edge_weight, edge_weight)
            wbce = tf.keras.losses.BinaryCrossentropy()(targets, outputs, sample_weight=weights)
            metrics['wbce'] = float(wbce.numpy())

        with open(os.path.join(eval_dir, 'metrics.yml'), 'w') as yaml_file:
            yaml.dump(metrics, yaml_file)

        b, t, h, w, c = inputs.shape

        for i in range(amount):
            series_inputs = _get_input_videos_as_images(inputs[i], split_timestep, sensors_in, channels_list)
            series_target = _get_video_as_image(targets[i], split_timestep)
            series_output = _get_video_as_image(outputs[i], split_timestep)
            series_combined = tf.concat(series_inputs + [series_target, series_output], axis=0)
            tf.keras.utils.save_img(os.path.join(eval_dir, f'result_{i}.png'), series_combined)

            fig, ax = plt.subplots()
            plt.imshow(series_combined)

            plt.xticks([(j*w+w//2) if j < split_timestep else (j*w+w//2+w//16) for j in range(t)])
            ax.set_xticklabels(range(1, t+1))
            ax.xaxis.set_tick_params(length=0)

            plt.yticks([(j*h+h//2) for j in range(len(sensors_in)+2)])
            ax.set_yticklabels(len(sensors_in)*[''] + ['target', 'output'])
            ax.yaxis.set_tick_params(length=0)

            plt.rcParams['svg.fonttype'] = 'none'
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.savefig(os.path.join(eval_dir, f'figure_{i}.svg'), bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
            plt.close()

            df = pd.DataFrame.from_dict({
                'psnr': psnrs[i].numpy(),
                'ssim': ssims[i].numpy(),
                'speed': speed[i]
            })

            df.to_csv(os.path.join(eval_dir, f'metrics_{i}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description.')
    parser.add_argument('directory', help='Model directory containing config.yml file.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch')
    parser.add_argument('--amount', type=int, default=10, help='Amount')
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"Directory does not exit: {args.directory}")
        raise ValueError

    plot_results_model_dir(args.directory, args.epoch, amount=args.amount)
