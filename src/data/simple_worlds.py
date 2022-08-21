"""
Author of file: Manuel Woellhaf
"""
import math
import numpy as np
from scipy.stats import truncnorm


class simple_2d:

    def __init__(
            self,
            seq_len=20,
            distribution='deterministic',
            label=None,
            limits=(-8, 8),
            speed=1.0,
            it_length=int(10e6),
            seed=0
    ):
        """
        World with two dimensional static state and four dimensional dynamic state.
        """
        self._seq_len = seq_len
        self._limits = np.array([limits, limits], dtype=np.float32)
        self._speed = speed
        self._distribution = distribution
        self._label = label

        self.shape = (self._seq_len, 2)
        self._length = it_length
        self.random = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __len__(self):
        return self._length

    def _angle_limits(self, out_of_limits):
        """
        If limits where surpassed in two dimension within this step,
        we restrict the angle to pi/2. This is a little inconsistend
        but avoids looping in a corner and increases performance.
        """
        xl, xu, yl, yu = out_of_limits[0, 0], out_of_limits[0, 1], out_of_limits[1, 0], out_of_limits[1, 1]
        x = xl or xu
        y = yl or yu

        if xl and not y:
            return [-np.pi/2, np.pi/2]
        elif xu and not y:
            return [np.pi/2, 3*np.pi/2]
        elif yl and not x:
            return [0, np.pi]
        elif yu and not x:
            return [-np.pi, 0]
        elif xl and yl:
            return [0, np.pi/2]
        elif xl and yu:
            return [-np.pi/2, 0]
        elif xu and yl:
            return [np.pi/2, np.pi]
        elif xu and yu:
            return [-np.pi, -np.pi/2]

    def __next__(self, *args, **kwargs):  # init_pos=None, init_vel=None):
        """
        init_pos: [2,], [0, 1)
        init_vel: [2,], [0, 1)
        """
        if args:
            self.random = np.random.RandomState(args[0])

        position = np.zeros((self._seq_len, 2), dtype=np.float32)
        if 'init_pos' in kwargs:
            position[0] = kwargs['init_pos']
        else:
            position[0] = self.random.rand(2)*(self._limits[:, 1]-self._limits[:, 0])+self._limits[:, 0]

        velocity = np.zeros((self._seq_len, 2), dtype=np.float32)
        if 'init_vel' in kwargs:
            velocity[0] = kwargs['init_vel']
        else:
            angle = np.pi * (self.random.rand()*2 - 1)
            velocity[0] = np.array([self._speed*math.cos(angle), self._speed*math.sin(angle)])

        for t in range(self._seq_len-1):
            cur_vel = velocity[t].copy()
            cur_pos = position[t].copy()
            cur_dst = 0
            speed = np.linalg.norm(cur_vel)
            new_pos = cur_pos + cur_vel

            # if we hit the limits we have to change direction
            lower, upper = new_pos < self._limits[:, 0], self._limits[:, 1] < new_pos
            out_of_limits = np.transpose(np.stack([lower, upper]))
            while np.any(out_of_limits) and speed > cur_dst:

                eps = 10e-8
                # distance to limits relative to velocity
                rd = abs((self._limits.T-cur_pos)/(cur_vel+eps)).T
                impact_lim = rd == min(rd[out_of_limits])
                rd = rd[impact_lim][0]

                # walk until we hit limits and keep track of distance within one step
                cur_pos += rd*(cur_vel+eps)
                cur_dst += rd

                # generate new velocity
                if self._distribution == 'deterministic':
                    flip = lower + upper
                    cur_vel = cur_vel*(np.ones(cur_vel.shape, dtype=np.int8)-2*flip)

                elif self._distribution == 'binomial':
                    flip = lower + upper
                    cur_vel = cur_vel*(np.ones(cur_vel.shape, dtype=np.int8)-2*flip)
                    if self.random.rand() >= 0.5:
                        cur_vel = cur_vel*(np.ones(cur_vel.shape, dtype=np.int8)-2*~flip)

                elif self._distribution == 'uniform':
                    alims = self._angle_limits(impact_lim)
                    angle = self.random.rand()*(alims[1]-alims[0])+alims[0]
                    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])

                elif self._distribution == 'gaussian':
                    flip = lower + upper
                    cur_vel = cur_vel*(np.ones(cur_vel.shape, dtype=np.int8)-2*flip)
                    alims = self._angle_limits(impact_lim)
                    angle = np.arctan2(cur_vel[1], cur_vel[0])
                    if not (alims[0] <= angle <= alims[1]):
                        angle += 2*np.pi
                    # angle = self.random.normal(loc=angle, scale=0.2)
                    # angle = np.clip(angle, alims[0], alims[1])
                    angle = truncnorm((alims[0]-angle)/0.2, (alims[1]-angle)/0.2, loc=angle, scale=0.2).rvs()
                    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])

                else:
                    raise NotImplementedError('Simple2D distribution: '+self._distribution)

                # walk with new velocity whatever distance is left for this step
                new_pos = cur_pos + max(0, speed-cur_dst)*cur_vel

                # if we hit the limits again, we have to change direction again
                # it's fine though if we land on the limits as step is finished
                lower, upper = new_pos < self._limits[:, 0], self._limits[:, 1] < new_pos
                out_of_limits = np.transpose(np.stack([lower, upper]))

            position[t+1] = new_pos
            velocity[t+1] = cur_vel

        assert np.all(self._limits[:, 0] <= position) and np.all(position <= self._limits[:, 1])
        features = np.concatenate([position, velocity], axis=-1)

        if self._label is None:
            return features

        # if self._label == 'features':
        #     return (features, features)

        if self._label == 'duplicate':
            return (features, features)

        if self._label.split('@')[0] == 'duplicate':
            split_index = int(self._label.split('@')[1])
            return (features, features[split_index:])

        if self._label == 'split':
            split_index = self._seq_len//2
            return (features[:split_index], features[split_index:])

        if self._label.split('@')[0] == 'split':
            split_index = int(self._label.split('@')[1])
            return (features[:split_index], features[split_index:])


def get_dataset_name(distribution, pown, seq_len, version, size, dim):
    return 'simple2d' + \
           '_' + distribution[:3] + \
           '_n' + str(pown) + \
           '_l' + str(seq_len) + \
           '_v' + str(version) + \
           '_s' + str(size) + \
           '_' + str(dim) + 'd'


def generate_simple_2d_dataset(distribution, seq_len=20, pown=7):
    n = int(10**pown)
    dobj = simple_2d(seq_len=seq_len, distribution=distribution)

    data = np.zeros((n, seq_len, 4), dtype=np.float32)
    print('generate ...')
    from tqdm import tqdm
    for i in tqdm(range(n)):
        data[i] = dobj.__next__()

    print('test ...')
    unique = np.unique(data, axis=0)
    assert unique.shape[0] == data.shape[0]

    eval_batch = generate_evaluation_batch(distribution, seq_len, nsamples=1)
    print(data.shape, eval_batch.shape)
    data = np.concatenate([data, eval_batch])
    print(data.shape, eval_batch.shape)

    np.save(get_dataset_name(distribution, pown, seq_len, 1, 16, 4), data)
    np.save(get_dataset_name(distribution, pown, seq_len, 1, 16, 2), data[..., :2])


def generate_evaluation_batch(distribution, seq_len=20, nsamples=1):
    init_pos = (-8, -8)

    dobj = simple_2d(seq_len=seq_len, distribution=distribution)

    runs = []

    angle, speed = math.radians(11.25), 1.0
    pos = np.array(init_pos) + (16/math.cos(angle)-12)/math.sqrt(2)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    runs.append(np.stack([dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]))

    angle, speed = math.radians(22.50), 1.0
    pos = np.array(init_pos) + (16/math.cos(angle)-12)/math.sqrt(2)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    runs.append(np.stack([dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]))

    angle, speed = math.radians(45.00), 1.0
    pos = np.array(init_pos) + (16/math.cos(angle)-12)/math.sqrt(2)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    runs.append(np.stack([dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]))

    angle, speed = math.radians(67.50), 1.0
    pos = np.array(init_pos) + (16/math.sin(angle)-12)/math.sqrt(2)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    runs.append(np.stack([dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]))

    angle, speed = math.radians(0), 1.0
    pos = (-4, 0)
    cur_vel = np.array([speed*math.cos(angle), speed*math.sin(angle)])
    runs.append(np.stack([dobj.__next__(init_pos=pos, init_vel=cur_vel) for _ in range(nsamples)]))

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(len(runs), seq_len, sharey=True, sharex=True, figsize=(seq_len, len(runs)))
    # for i in range(len(runs)):
    #     for t in range(seq_len):
    #         axes[i, t].set_ylim(limits)
    #         axes[i, t].set_xlim(limits)
    #         obj = axes[i, t].scatter(runs[i][:, t, 0], runs[i][:, t, 1], marker='x', c='r')
    # plt.show()
    return np.squeeze(np.stack(runs))


if __name__ == '__main__':

    generate_simple_2d_dataset('binomial', seq_len=20, pown=6)
    # generate_evaluation_batch('deterministic')
    # import matplotlib.pyplot as plt

    # length = 20
    # limits = (-8, 8)

    # nsamples = 10
    # runs = []

    # import time

    # def timeit(ds, steps=5):

    #     start = time.time()
    #     it = iter(ds)
    #     for i in range(steps):
    #         next(it)

    #     end = time.time()
    #     duration = end-start

    #     print("{} batches: {} s".format(steps, duration))
    #     print("{:0.5f} Images/s".format(steps/duration))

    # dobj = simple_2d(seq_len=length, distribution='gaussian', label=None, limits=limits)
    # # dobj = simple_2d(seq_len=length, distribution='binomial', label=None, limits=limits)
    # # dobj = simple_2d(seq_len=length, distribution='uniform', label=None, limits=limits)
    # # dobj = simple_2d(seq_len=length, distribution='deterministic', label=None, limits=limits)
    # # dobj(0)
    # timeit(dobj, 1000)

    # dobj = simple_2d(seq_len=length, distribution='deterministic', label=None, limits=limits)
    # samples = [dobj.__next__(init_pos=(-8, 8), init_vel=(1, -0.5)) for i in range(nsamples)]
    # runs.append(np.stack(samples))

    # dobj = simple_2d(seq_len=length, distribution='binomial', label=None, limits=limits)
    # samples = [dobj.__next__(init_pos=(-8, 8), init_vel=(1, -0.5)) for i in range(nsamples)]
    # runs.append(np.stack(samples))

    # dobj = simple_2d(seq_len=length, distribution='gaussian', label=None, limits=limits)
    # samples = [dobj.__next__(init_pos=(-8, 8), init_vel=(1, -0.5)) for i in range(nsamples)]
    # runs.append(np.stack(samples))

    # dobj = simple_2d(seq_len=length, distribution='uniform', label=None, limits=limits)
    # samples = [dobj.__next__(init_pos=(-8, 8), init_vel=(1, -0.5)) for i in range(nsamples)]
    # runs.append(np.stack(samples))

    # dobj = simple_2d(seq_len=length, distribution='deterministic', label=None, limits=limits)
    # samples = [dobj.__next__(init_pos=(-8, 8), init_vel=(1, -1)) for i in range(nsamples)]
    # runs.append(np.stack(samples))

    # dobj = simple_2d(seq_len=length, distribution='binomial', label=None, limits=limits)
    # samples = [dobj.__next__(init_pos=(-8, 8), init_vel=(1, -1)) for i in range(nsamples)]
    # runs.append(np.stack(samples))

    # dobj = simple_2d(seq_len=length, distribution='gaussian', label=None, limits=limits)
    # samples = [dobj.__next__(init_pos=(-8, 8), init_vel=(1, -1)) for i in range(nsamples)]
    # runs.append(np.stack(samples))

    # dobj = simple_2d(seq_len=length, distribution='uniform', label=None, limits=limits)
    # samples = [dobj.__next__(init_pos=(-8, 8), init_vel=(1, -1)) for i in range(nsamples)]
    # runs.append(np.stack(samples))

    # print(runs[0].shape)

    # fig, axes = plt.subplots(len(runs), length, sharey=True, sharex=True, figsize=(length, len(runs)))
    # for i in range(len(runs)):
    #     for t in range(length):
    #         axes[i, t].set_ylim(limits)
    #         axes[i, t].set_xlim(limits)
    #         obj = axes[i, t].scatter(runs[i][:, t, 0], runs[i][:, t, 1], c='r')

    # plt.show()
