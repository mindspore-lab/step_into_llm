import mindspore
from mindspore import ops, nn, Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
import numpy as np

class BertLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = nn.WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = nn.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = ops.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = ops.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mindspore.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr

class WarmUpPolynomialDecayLR(LearningRateSchedule):
    """"""
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.end_learning_rate = end_learning_rate
        self.decay_steps = decay_steps
        self.power = power
    
    def construct(self, global_step):
        # warmup lr
        warmup_percent = global_step.astype(mindspore.float32) / self.warmup_steps
        warmup_learning_rate = self.learning_rate * warmup_percent
        # polynomial lr
        global_step = ops.minimum(global_step, self.decay_steps)
        decayed_learning_rate = (self.learning_rate - self.end_learning_rate) * \
                                ops.pow((1 - global_step / self.decay_steps), self.power) + \
                                self.end_learning_rate
        is_warmup = (global_step < self.warmup_steps).astype(mindspore.float32)
        learning_rate = ((1.0 - is_warmup) * decayed_learning_rate + is_warmup * warmup_learning_rate)
        return learning_rate

def create_optimizer(model, init_lr, num_train_steps, num_warmup_steps):
    # lr = WarmUpPolynomialDecayLR(init_lr, 0.0, num_warmup_steps, num_train_steps, 1.0)
    lr = BertLearningRate(init_lr, 0.0, num_warmup_steps, num_train_steps, 1.0)
    params = model.trainable_params()
    decay_params = list(filter(lambda x: 'gamma' not in x.name \
                                        and 'beta' not in x.name \
                                        and 'bias' not in x.name, params))
    other_params = list(filter(lambda x: 'gamma' in x.name \
                                        or 'beta' in x.name \
                                        or 'bias' in x.name, params))
    group_params = [{'params': decay_params, 'weight_decay': 0.01},
                    {'params': other_params},
                    {'order_params': params}]
    # optim = nn.AdamWeightDecay(group_params, lr)
    optim = nn.Lamb(group_params, learning_rate=lr, beta1=0.99, beta2=0.999,
                            eps=1e-8)
    return optim