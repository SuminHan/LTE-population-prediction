import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


def get_conv_layer(conv_type: str = "standard") -> torch.nn.Module:
    if conv_type == "standard":
        conv_layer = tf.keras.layers.Conv2D
    elif conv_type == "coord":
        conv_layer = CoordConv
    elif conv_type == "3d":
        conv_layer = tf.keras.layers.Conv3D
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv_layer


# https://github.com/thisisiron/spectral_normalization-tf2/blob/master/sn.py
class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output
    
    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)


class DBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        output_channels: int = 12,
        conv_type: str = "standard",
        first_relu: bool = True,
        keep_same_output: bool = False,
    ):
        """
        D and 3D Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            # input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Convolution type, see satflow/models/utils.py for options
            first_relu: Whether to have an ReLU before the first 3x3 convolution
            keep_same_output: Whether the output should have the same spatial dimensions as input, if False, downscales by 2
        """
        super().__init__()
        self.output_channels = output_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        conv2d = get_conv_layer(conv_type)
        if conv_type == "3d":
            # 3D Average pooling
            self.pooling = torch.nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def build(self):
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            )
        )
        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )
        # Downsample at end of 3x3
        self.relu = torch.nn.ReLU()
        # Concatenate to double final channels and keep reduced spatial extent

    def call(self, x):
        if self.input_channels != self.output_channels:
            x1 = self.conv_1x1(x)
            if not self.keep_same_output:
                x1 = self.pooling(x1)
        else:
            x1 = x

        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.last_conv_3x3(x)

        if not self.keep_same_output:
            x = self.pooling(x)
        x = x1 + x  # Sum the outputs should be half spatial and double channels
        return x



class MyGenerator(Model):
    def __init__(self, args, metadata):
        super(MyGenerator, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyGenerator'
        
    def build(self, input_shape):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.latent_dim,)),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(784, activation='tanh'),
            layers.Reshape((28, 28))
        ])

    def call(self, context):
        """Connect to a graph.
        Args:
        inputs: a batch of inputs on the shape [batch_size, P, h, w, 1].
        Returns:
        predictions: a batch of predictions in the form
            [batch_size, Q, h, w, 1].
        """

        x = tf.transpose(context, (0, 2, 3, 1, 4))
        x = tf.reshape(x, (-1, self.height, self.width, self.P))



        
        return predictions




class MyDiscriminator(Model):
    def __init__(self):
        super(MyDiscriminator, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(28, 28)),
            layers.Reshape((784,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def call(self, inputs):
        return self.model(inputs)

class GAN(Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.generator.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise)

            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(fake_images)

            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}
