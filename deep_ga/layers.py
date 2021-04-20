from tensorflow import keras
import tensorflow as tf
import keras.backend as K
import numpy as np

# Instead of having a normal 3x3 kernel, this conv layer is forced to
# have a kernel that only performs differentiations, i.e. the kernel
# has the following shape:
#                            A  B  C
#                            D  0 -D
#                           -C -B -A
#


class SymConv2D(keras.layers.Layer):
    def __init__(self, filters, sobelInitial=False, ** kwargs):
        self.filters = filters
        self.sobelInitial = sobelInitial
        super(SymConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[-1] == 1, "Only implemented for one input"
        shapew = (4,) + (input_shape[-1], self.filters)
        if self.sobelInitial:
            assert self.filters < 5, "Only four sobel filters available"
            sobel_weights = np.array([
                [1, 3, 1, 0],  # x
                [1, 0, -1, 3],  # y
                [3, 1, 0, 1],  # xy
                [0, 1, 3, -1]  # yx
            ], np.float32)
            sobel_weights = np.reshape(sobel_weights.transpose(), (4, 1, 4))
            sobel_weights = sobel_weights[:, :, :self.filters]
            sobel_weights = np.repeat(sobel_weights, input_shape[-1], axis=1)
            self.w = tf.Variable(initial_value=sobel_weights)
        else:
            self.w = self.add_weight(name='kernel', shape=shapew,
                                     initializer='glorot_uniform')
        super(SymConv2D, self).build(input_shape)

    def call(self, x):
        # duplicate rows 0 and 2
        old_opts = tf.config.optimizer.get_experimental_options()
        tf.config.optimizer.set_experimental_options(
            {'layout_optimizer': False})

        isnan = tf.math.is_nan(x)
        isnan = tf.cast(isnan, tf.uint8)
        isnan = tf.nn.max_pool(isnan, ksize=3, strides=1, padding="VALID")
        isnan = tf.cast(isnan, tf.bool)

        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

        r0 = K.stack((self.w[0], self.w[1], self.w[2]), axis=0)
        r1 = K.stack((self.w[3], K.zeros_like(self.w[0]), -self.w[3]), axis=0)
        r2 = K.stack((-self.w[2], -self.w[1], -self.w[0]), axis=0)
        kernel = K.stack((r0, r1, r2), axis=1)

        conv = K.conv2d(x, kernel)
        conv = tf.where(isnan, tf.zeros_like(conv), conv)

        tf.config.optimizer.set_experimental_options(old_opts)
        return conv

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'sobelInitial': self.sobelInitial
        })
        return config


class PConv2D(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(PConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        shapew = (9,) + (input_shape[-1], self.filters)
        self.w = self.add_weight(name='kernel', shape=shapew,
                                      initializer='glorot_uniform')
        super(PConv2D, self).build(input_shape)

    def call(self, x):
        isnan = tf.math.is_nan(x)
        isnan = tf.cast(isnan, tf.uint8)
        isnan = tf.nn.max_pool(isnan, ksize=3, strides=1, padding="VALID")
        isnan = tf.cast(isnan, tf.bool)

        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

        r0 = K.stack((self.w[0], self.w[1], self.w[2]), axis=0)
        r1 = K.stack((self.w[3], self.w[4], self.w[5]), axis=0)
        r2 = K.stack((self.w[6], self.w[7], self.w[8]), axis=0)
        kernel = K.stack((r0, r1, r2), axis=1)

        conv = K.conv2d(x, kernel)
        conv = tf.where(isnan, tf.zeros_like(conv), conv)
        return conv

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters
        })
        return config


class NanToZero(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NanToZero, self).__init__(**kwargs)

    def call(self, x):
        return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    def get_config(self):
        config = super().get_config().copy()
        return config


class IsNanMask(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(IsNanMask, self).__init__(**kwargs)

    def call(self, x):
        return tf.cast(tf.math.is_nan(x), tf.float32)

    def get_config(self):
        config = super().get_config().copy()
        return config


# This layer computes the euclidean distance between two vectors
class EuclideanDistanceLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)

    def call(self, x):
        return K.sqrt(K.sum(K.square(x[1] - x[0]), axis=-1, keepdims=True))

    def get_config(self):
        config = super().get_config().copy()
        return config


import losses

custom_objects = {
    "SymConv2D": SymConv2D,
    "PConv2D": PConv2D,
    "EuclideanDistanceLayer": EuclideanDistanceLayer,
    "NanToZero": NanToZero,
    "pairwise_contrastive_loss": losses.pairwise_contrastive_loss,
    "binary_cross_entropy": losses.binary_cross_entropy,
    "doomloss": losses.doomloss
}
