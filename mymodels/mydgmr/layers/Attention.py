import tensorflow as tf

def attention_einsum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""

    # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
    k = tf.reshape(k, [-1, k.shape[-1]])  # [h, w, c] -> [L, c]
    v = tf.reshape(v, [-1, v.shape[-1]])  # [h, w, c] -> [L, c]

    # Einstein summation corresponding to the query * key operation.
    beta = tf.nn.softmax(tf.einsum("hwc, Lc->hwL", q, k), axis=-1)

    # Einstein summation corresponding to the attention * value operation.
    out = tf.einsum("hwL, Lc->hwc", beta, v)
    return out

class AttentionLayer(tf.keras.layers.Layer):
    """Attention Module"""

    def __init__(self, input_channels: int, output_channels: int, ratio_kq=8, ratio_v=8):
        super(AttentionLayer, self).__init__()

        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.output_channels = output_channels
        self.input_channels = input_channels

        # Compute query, key and value using 1x1 convolutions.
        self.query = tf.keras.layers.Conv2D(
            filters=self.output_channels // self.ratio_kq,
            kernel_size=(1, 1),
            padding="valid",
            use_bias=False,
        )
        self.key = tf.keras.layers.Conv2D(
            filters=self.output_channels // self.ratio_kq,
            kernel_size=(1, 1),
            padding="valid",
            use_bias=False,
        )
        self.value = tf.keras.layers.Conv2D(
            filters=self.output_channels // self.ratio_v,
            kernel_size=(1, 1),
            padding="valid",
            use_bias=False,
        )

        self.last_conv = tf.keras.layers.Conv2D(
            filters=self.output_channels,
            kernel_size=(1, 1),
            padding="valid",
            use_bias=False,
        )

        # Learnable gain parameter
        self.gamma = self.add_weight(
            shape=[1],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="gamma",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        # Compute query, key and value using 1x1 convolutions.
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Apply the attention operation.
        out = []
        for b in range(batch_size):
            # Apply to each in batch
            out.append(attention_einsum(query[b], key[b], value[b]))
        out = tf.stack(out, axis=0)
        out = self.gamma * self.last_conv(out)
        # Residual connection.
        return out + x
