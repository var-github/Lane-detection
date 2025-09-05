from tensorflow import reduce_mean, reduce_max, concat, expand_dims
from tensorflow.keras import layers, losses
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def weighted_bce(y_true, y_pred):
    penality = 20.0
    # y_true = 1 when it is a lane pixel, then w = 50
    # y_true = 0 when it background, then w = 1
    w = y_true * penality + (1 - y_true) * 1.0

    # Compute standard binary cross-entropy
    bce = losses.binary_crossentropy(y_true, y_pred)
    # original image shape - [batch, height, width, 1], bce shape = [batch, height, width]
    # Hence adding another dimension
    bce = expand_dims(bce, axis=-1)

    # .reduce_mean takes matrix like this [[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]] and averages all numbers
    # Here we are taking mean of error of each pixel
    return reduce_mean(w * bce)


@register_keras_serializable()
class spatial_attention(layers.Layer):
    def __init__(self, **kwargs):
        super(spatial_attention, self).__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = reduce_max(inputs, axis=3, keepdims=True)
        join = concat([avg_pool, max_pool], axis=3)
        attention = self.conv(join)
        return inputs * attention

    def get_config(self):
        base_config = super(spatial_attention, self).get_config()
        return base_config
