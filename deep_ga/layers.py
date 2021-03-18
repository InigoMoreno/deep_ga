from tensorflow import keras
import keras.backend as K

# Instead of having a normal 3x3 kernel, this conv layer is forced to
# have a kernel that only performs differentiations, i.e. the kernel
# has the following shape:
#                            A  B  C
#                            D  0 -D
#                           -C -B -A
#


class SymConv2D(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(SymConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        shapew = (4,) + (input_shape[-1], self.filters)
        self.w = self.add_weight(name='kernel', shape=shapew,
                                      initializer='glorot_uniform')
        super(SymConv2D, self).build(input_shape)

    def call(self, x):
        # duplicate rows 0 and 2
        r0 = K.stack((self.w[0], self.w[1], self.w[2]), axis=0)
        r1 = K.stack((self.w[3], K.zeros_like(self.w[0]), -self.w[3]), axis=0)
        r2 = K.stack((-self.w[2], -self.w[1], -self.w[0]), axis=0)
        kernel = K.stack((r0, r1, r2), axis=1)

        return K.conv2d(x, kernel)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters
        })
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
