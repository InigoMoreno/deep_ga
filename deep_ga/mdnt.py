################################################################
# Optimizers - Phase-mixed optimizers.
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# This module contains the optimizers that has multiple phases.
# In different phases, those optimizers would adopt different
# algorithms. A typical example is the SWATS optimizer.
################################################################

from tensorflow.python.framework import ops
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops


def m_switch(pred, tensor_a, tensor_b):
    '''
    Use cleaner API to replace m_switch to accelerate computation.
    '''
    def f_true(): return tensor_a
    def f_false(): return tensor_b
    return control_flow_ops.cond(pred, f_true, f_false, strict=True)


class SWATS(optimizers.Optimizer):
    """Switches from Adam to SGD (SWATS)
    From Adam optimizer to SGD optimizer automatically.
    This method provides an automatic scheme for switching from Adam/Amsgrad to
    SGD optimizer. Different from manually switched optimizer. The switching point
    where Adam switches to SGD is determined by the algorithm, so users do not
    need to call the switch notifier or use manually switching callback.
    The key technique for this method is checking the convergence of the relation-
    ship between the adaptive momentum and the SGD gradient. If the proportion of
    the gradient projected in the direction of the momentum becomes converged, the
    algorithm would switch to SGD silently.
    One difference between SWATS and manually switched optimizers is that SWATS
    determine the algorithm phases for each optimized parameter independently, i.e.
    different parameters may be optimized in different phases.
    Another difference is that the SWATS would determine the learning rate of SGD
    automatically to ensure that the learning rate would not change drastically
    after the switching point.
    This implementation has such modifications compared to the original work:
        1. The manually set learning rate is separated from lambda to make sure
           that users could modify learning rate after switching to SGD.
        2. The finally converged lambda is restricted to > 0, otherwise it will
           not switch from Adam to SGD.
        3. It supports Amsgrad -> SGD.
    Due to the limitation of the tensorflow, this implementation is not highly
    efficient. Users should estimate whether it is necessary to use this optimi-
    zer.
    Arguments:
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    Reference:
        This optimizer is derived from this paper:
            Improving Generalization Performance by Switching from Adam to SGD
            https://arxiv.org/abs/1712.07628
        An unofficial implementation which inspires this work could be referred
        here:
            https://github.com/sloth2012/scalaML
    """

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 amsgrad=False,
                 **kwargs):
        super(SWATS, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay *
                       math_ops.cast(self.iterations, K.dtype(self.decay))))

        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
            t = math_ops.cast(self.iterations, K.floatx())
        lr_bc = gen_math_ops.sqrt(
            1. - math_ops.pow(self.beta_2, t)) / (1. - math_ops.pow(self.beta_1, t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        lams = [K.zeros(1, dtype=K.dtype(p)) for p in params]
        conds = [K.variable(False, dtype='bool') for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats + lams + conds

        for p, g, m, v, vhat, lam, cond in zip(params, grads, ms, vs, vhats, lams, conds):
            beta_g = m_switch(cond, 1.0, 1.0 - self.beta_1)
            m_t = (self.beta_1 * m) + beta_g * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsgrad:
                vhat_t = math_ops.maximum(vhat, v_t)
                p_t_ada = lr_bc * m_t / \
                    (gen_math_ops.sqrt(vhat_t) + self.epsilon)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                p_t_ada = lr_bc * m_t / (gen_math_ops.sqrt(v_t) + self.epsilon)
            gamma_den = math_ops.reduce_sum(p_t_ada * g)
            gamma = math_ops.reduce_sum(gen_math_ops.square(p_t_ada)) / (math_ops.abs(
                gamma_den) + self.epsilon) * (gen_math_ops.sign(gamma_den) + self.epsilon)
            lam_t = (self.beta_2 * lam) + (1. - self.beta_2) * gamma
            lam_prime = lam / (1. - math_ops.pow(self.beta_2, t))
            lam_t_prime = lam_t / (1. - math_ops.pow(self.beta_2, t))
            lg_err = math_ops.abs(lam_t_prime - gamma)
            cond_update = gen_math_ops.logical_or(gen_math_ops.logical_and(
                gen_math_ops.logical_and(self.iterations > 1, lg_err < 1e-5), lam_t > 0), cond)[0]
            lam_update = m_switch(cond_update, lam, lam_t)
            self.updates.append(state_ops.assign(lam, lam_update))
            self.updates.append(state_ops.assign(cond, cond_update))

            p_t_sgd = (1. - self.beta_1) * lam_prime * m_t

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))

            new_p = m_switch(cond, p - lr * p_t_sgd, p - lr * p_t_ada)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        }
        base_config = super(SWATS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
