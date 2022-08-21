"""
Author of file: Manuel Woellhaf
"""
import math
import numpy as np

from .simple_worlds import simple_2d


def draw_whit_dot(frame, radius=3):
    nx, ny = frame.shape[0], frame.shape[1]
    cx, cy = (nx+1)//2, (ny+1)//2
    y, x = np.ogrid[-cx:nx-cx, -cy:ny-cy]
    mask = x*x + y*y <= radius*radius
    frame[mask] = np.iinfo(frame.dtype).max


class whit_dots:

    def __init__(
            self,
            shape=32,
            seq_len=20,
            dots=1,
            radius=3,
            distribution='deterministic',
            speed=1,
            label=None,
            it_length=1000,
            seed=None
    ):
        """
        """
        self._seq_len = seq_len
        self._dots = dots
        self._r = radius
        self._distribution = distribution
        self._label = label
        self._speed = speed

        self._dot = np.zeros([2*radius+1, 2*radius+1, 1], dtype=np.uint16)
        draw_whit_dot(self._dot, radius)

        self.shape = (self._seq_len, shape, shape, 1)

        self._limits = ((self._dot.shape[0]+1)//2, shape - (self._dot.shape[0]+1)//2)
        self._s2d = simple_2d(self._seq_len, self._distribution, label=None, limits=self._limits, speed=self._speed)

        self.random = np.random.RandomState(seed)
        self._length = it_length

    def __iter__(self):
        return self

    def __len__(self):
        return self._length

    def __next__(self, *args, **kwargs):
        """
        init_pos: [dots, dims], [0, 1)
        init_vel: [dots, dims], [0, 1)
        """
        sequence = np.zeros(self.shape, dtype=np.uint16)

        if 'init_pos' in kwargs:
            features = [
                self._s2d.__next__(
                    init_pos=kwargs['init_pos'][i], init_vel=kwargs['init_vel'][i]
                )
                for i in range(self._dots)]
        else:
            features = [self._s2d.__next__() for i in range(self._dots)]

        for frame_idx in range(self._seq_len):
            for i in range(self._dots):
                pos = features[i][frame_idx, :2].astype(np.uint8)
                s = pos - np.array(self._dot.shape[:2], dtype=np.uint8)//2
                e = pos + (np.array(self._dot.shape[:2], dtype=np.uint8)+1)//2
                sequence[frame_idx, s[0]:e[0], s[1]:e[1]] += self._dot

        sequence = np.clip(sequence, 0, np.iinfo(np.uint8).max).astype(np.float32)/np.iinfo(np.uint8).max

        if self._label is None:
            return sequence, np.array(features)

        if self._label == 'duplicate':
            return (sequence, sequence)

        if self._label.split('@')[0] == 'duplicate':
            split_index = int(self._label.split('@')[1])
            return (sequence, sequence[split_index:])

        if self._label == 'split':
            split_index = self._seq_len//2
            return (sequence[:split_index], sequence[split_index:])

        if self._label.split('@')[0] == 'split':
            split_index = int(self._label.split('@')[1])
            return (sequence[:split_index], sequence[split_index:])


def get_dataset_name(distribution, pown, seq_len, version, shape, dots):
    return 'whit_dots' + \
           '_' + distribution[:3] + \
           '_n' + str(pown) + \
           '_l' + str(seq_len) + \
           '_v' + str(version) + \
           '_s' + str(shape) + \
           '_d' + str(dots)

        
def generate_whit_dots_dataset(distribution, dots=1, shape=32, seq_len=20, pown=7):
    n = int(10**pown)
    dobj = whit_dots(shape=shape, seq_len=seq_len, dots=dots, distribution=distribution)

    data = np.zeros((n, seq_len, shape, shape, 1), dtype=np.float16)
    labels = np.zeros((n, seq_len, 4), dtype=np.float32)
    print('generate ...')
    from tqdm import tqdm
    for i in tqdm(range(n)):
        data[i], labels[i] = dobj.__next__()

    print('test ...')
    unique = np.unique(labels, axis=0)
    assert unique.shape[0] == labels.shape[0]

    eval_batch = generate_evaluation_batch(distribution, dots, shape, seq_len, nsamples=1)
    print(data.shape, eval_batch[0].shape)
    data = np.concatenate([data, np.expand_dims(eval_batch[0], axis=-1)])
    labels = np.concatenate([labels, eval_batch[1]])
    print(data.shape, labels.shape)

    string = get_dataset_name(distribution, pown, seq_len, 1, shape, dots)
    np.save(string + '_data', data)
    np.save(string + '_labels', labels)


def generate_evaluation_batch(distribution, dots, shape, seq_len, nsamples=1):
    init_pos = (8, 8)

    dobj = whit_dots(dots=dots, shape=shape, seq_len=seq_len, distribution=distribution)

    runs = []

    angle, speed = math.radians(11.25), 1.0
    pos = np.array(init_pos) + (16/math.cos(angle)-12)/math.sqrt(2)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    result = [dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]
    runs.append((np.stack([r[0] for r in result]), np.stack([r[1] for r in result])))

    angle, speed = math.radians(22.50), 1.0
    pos = np.array(init_pos) + (16/math.cos(angle)-12)/math.sqrt(2)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    result = [dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]
    runs.append((np.stack([r[0] for r in result]), np.stack([r[1] for r in result])))

    angle, speed = math.radians(45.00), 1.0
    pos = np.array(init_pos) + (16/math.cos(angle)-12)/math.sqrt(2)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    result = [dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]
    runs.append((np.stack([r[0] for r in result]), np.stack([r[1] for r in result])))

    angle, speed = math.radians(67.50), 1.0
    pos = np.array(init_pos) + (16/math.sin(angle)-12)/math.sqrt(2)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    result = [dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]
    runs.append((np.stack([r[0] for r in result]), np.stack([r[1] for r in result])))

    angle, speed = math.radians(0), 1.0
    pos = (12, 0)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    result = [dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]
    runs.append((np.stack([r[0] for r in result]), np.stack([r[1] for r in result])))

    return np.squeeze(np.stack([r[0] for r in runs])), np.squeeze(np.stack([r[1] for r in runs]))


if __name__ == '__main__':
    generate_whit_dots_dataset('uniform', dots=1, shape=32, seq_len=20, pown=5)

    # import cProfile
    # import time
    # # from ..utils.visio import generate_video, sequence_to_image
    # # import matplotlib.pyplot as plt

    # # sequences, features = generate_evaluation_batch()
    # # for sample in sequences:
    # #     image = sequence_to_image(sample, sample, sample)
    # #     plt.imshow(np.squeeze(image), cmap='gray')
    # #     plt.show()

    # def timeit(ds, steps=5):

    #     start = time.time()
    #     it = iter(ds)
    #     for i in range(steps):
    #         next(it)

    #     end = time.time()
    #     duration = end-start

    #     print("{} batches: {} s".format(steps, duration))
    #     print("{:0.5f} Images/s".format(steps/duration))

    # dots = whit_dots(
    #     shape=32,
    #     seq_len=20,
    #     dots=1,
    #     radius=3,
    #     # distribution='deterministic',
    #     # distribution='binomial',
    #     distribution='gaussian',
    #     # distribution='uniform',
    #     seed=0
    # )

    # # for i in range(10):
    # #     batch = next(dots)
    # #     print(batch.dtype, batch.shape, batch.min(), batch.max())
    # #     image = sequence_to_image(batch, batch, batch)
    # #     plt.imshow(np.squeeze(image), cmap='gray')
    # #     plt.show()

    # pr = cProfile.Profile()
    # pr.enable()
    # timeit(dots, 100)
    # pr.disable()
    # pr.dump_stats('whit.profile')
