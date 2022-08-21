"""
Author of file: Manuel Woellhaf
"""
import math
import tensorflow as tf


class ConstLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.lr = params['LR']

    def __call__(self, step):
        return self.lr


class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ This is cheekily copied from tensorflow addons. """

    def __init__(
            self,
            initial_learning_rate,
            maximal_learning_rate,
            step_size,
            scale_fn=None,
            name="cyclical_learning_rate",
    ):
        """
        Step size is the number of steps until the slope changes. This
        means we arrive back at the initial learning rate (= one cycle)
        after 2*step_size.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.step_size = step_size
        self.scale_fn = lambda x: 1. if scale_fn is None else scale_fn
        self.name = name

    def __call__(self, step):
        cycle = tf.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        return self.initial_learning_rate + (
            self.maximal_learning_rate - self.initial_learning_rate
        ) * tf.maximum(0.0, (1 - x))*self.scale_fn(cycle)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.initial_learning_rate,
            "step_size": self.step_size,
            "scale_fn": self.scale_fn,
            "name": self.name
        }


class CosineDecayLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Cosine decay with restarts. """
    
    def __init__(
            self,
            minimal_learning_rate,
            maximal_learning_rate,
            step_size,
            t_mul=1.0,
            m_mul=1.0,
            linear_warmup=False,
            name="cosine_decay_learning_rate"
    ):
        super().__init__()
        self.min_lr = minimal_learning_rate
        self.max_lr = float(maximal_learning_rate)
        self.step_size = step_size
        self._t_mul = t_mul
        self._m_mul = m_mul
        self._cycle = 1
        self.start_reduce_cycle = 3 if linear_warmup else 2
        self.end_cur_cycle = self.step_size
        self.name = name
        self.lin_warmup = tf.convert_to_tensor(linear_warmup)

    def __call__(self, step):
        max_lr = tf.convert_to_tensor(self.max_lr)
        dtype = max_lr.dtype
        min_lr = tf.cast(self.min_lr, dtype)
        t_mul = tf.cast(self._t_mul, dtype)
        m_mul = tf.cast(self._m_mul, dtype)
        step = tf.cast(step, dtype)
        step_size = tf.cast(self.step_size, dtype)
        cosine_step = tf.cond(
            tf.math.logical_and(self.lin_warmup, tf.math.greater(step, step_size)), lambda: step - step_size, lambda: step
        )

        def linear_warmup():
            return min_lr + (max_lr - min_lr)*step/step_size

        def cosine_decay():

            def restart_count_const():
                return tf.math.floor(cosine_step/step_size)
            def cycle_fraction_const(restart_count):
                return cosine_step/step_size-restart_count
            def restart_count_growing():
                return tf.math.floor(tf.math.log(cosine_step/step_size*(t_mul-1.0)+1.0)/tf.math.log(t_mul))
            def cycle_fraction_growing(restart_count):
                rsum = (1.0-t_mul**restart_count)/(1.0-t_mul)
                return (cosine_step/step_size-rsum)/t_mul**restart_count

            restart_count = tf.cond(
                tf.math.equal(t_mul, 1.0), restart_count_const, restart_count_growing
            )
            cycle_fraction = tf.cond(
                tf.math.equal(t_mul, 1.0), lambda: cycle_fraction_const(restart_count), lambda: cycle_fraction_growing(restart_count)
            )

            cur_max_lr = max_lr*m_mul**restart_count
            delta = cur_max_lr - min_lr
            growth = 0.5*delta*(1+tf.math.cos(tf.constant(math.pi)*cycle_fraction))
            return self.min_lr + growth

        return tf.cond(
            tf.math.logical_and(self.lin_warmup, tf.math.greater_equal(step_size, step)), linear_warmup, cosine_decay
        )


    def get_config(self):
        return {
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "step_size": self.step_size,
            "scale_fn": self.scale_fn,
            "lin_warmup": self.lin_warmup,
            "name": self.name
        }


if __name__ == '__main__':
    lrs = CosineDecayLearningRate(1, 10.0, step_size=10, m_mul=0.75, t_mul=2.0, linear_warmup=True)
    values = [lrs(s).numpy() for s in range(100)]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(values)
    plt.savefig('lrs')
