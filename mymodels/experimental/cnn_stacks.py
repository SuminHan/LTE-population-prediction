import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D
from keras.models import Sequential
from utils.loss import *

class MyCNNStacks(tf.keras.models.Model):
    def __init__(self, args, metadata):
        super(MyCNNStacks, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.D = args.D
        self.kernel_size = (args.conv_kernel_size, args.conv_kernel_size)
        self.teacher = args.teacher
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyCNNStacks_{args.conv_kernel_size}_{self.K}'
        
        pos_encoding = []
        for j in range(self.height):
            for i in range(self.width):
                one_hot = tf.one_hot(j*self.width + i, depth=self.num_cells)
                pos_encoding.append(one_hot)
        pos_encoding = tf.stack(pos_encoding, 0)
        self.pos_encoding = tf.cast(tf.reshape(pos_encoding, (self.height, self.width, self.num_cells)), dtype=tf.float32)
        
        
    def build(self, input_shape):
        # if self.K > 1:
        #     self.encoder_stack = keras.models.Sequential([
        #         ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, padding='same') for _ in range(self.K-1)
        #     ])
        #     self.decoder_stack = keras.models.Sequential([
        #         ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, padding='same') for _ in range(self.K-1)
        #     ])
        self.cnn_stacks = Sequential([Conv2D(self.D, self.kernel_size, padding='same', activation='tanh') for _ in range(self.K)])

        self.embed_layer = layers.Dense(self.D, use_bias=False)
        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='leaky_relu'),
                                layers.Dense(self.D, use_bias=False, activation='leaky_relu'),
                                layers.Dense(1, use_bias=False)])
        

    def call(self, X, TE):
        # X.shape (batch_size, P, data_height, data_width, 1)
        # TE.shape (batch_size, P+Q, 2)
        assert TE.shape[1] == self.P + self.Q
        batch_size = tf.shape(X)[0]
        
        #TE = tf.cast(tf.concat((tf.one_hot(TE[..., 0], depth=7), tf.one_hot(TE[..., 1], depth=24)), -1), dtype=tf.float32)
        TE = tf.cast(tf.one_hot(TE[..., 0]*24 + TE[..., 1], depth=24*7), dtype=tf.float32)
        # TE.shape (batch_size, P+Q, 7+24)
        
        TE = tf.expand_dims(tf.expand_dims(TE, 2), 2)
        TE = tf.tile(TE, [1, 1, self.height, self.width, 1])
        # TE.shape (batch_size, P+Q, data_height, data_width, 31)
        
        pos_encoding = self.pos_encoding
        pos_encoding = tf.expand_dims(tf.expand_dims(pos_encoding, 0), 0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, self.P+self.Q, 1, 1, 1])
        TE_pos = tf.concat((TE, pos_encoding), -1)
        # TE_pos.shape (batch_size, P+Q, data_height, data_width, 24*7+height*width)
        TE_pos_P, TE_pos_Q = TE_pos[:, :self.P, ...], TE_pos[:, self.P:, ...]
        
        embX = tf.concat((X, TE_pos_P), -1)
        # embX.shape  (batch_size, P, data_height, data_width, 1+24*7+height*width)
        embX = self.embed_layer(embX)
        # embX.shape  (batch_size, P, data_height, data_width, D)

        self_outputs = self.output_layer(self.cnn_stacks(embX))
        
        if self.teacher > 0:
            loss = self.teacher * custom_mae_loss(X[:, 1:, ...], self_outputs[:, :-1, ...])
            self.add_weight(loss)
            
        output_value = self_outputs[:, -1:, ...]
        sequential_output = [output_value]

        if self.Q > 1:
            for j in range(self.Q):
                TEQj_enc = tf.expand_dims(TE_pos_Q[:, j, ...], 1)
                next_X = tf.concat((output_value, TEQj_enc), -1)
                start_token = self.embed_layer(next_X)

                output_value = self.output_layer(self.cnn_stacks(embX))
                sequential_output.append(output_value)
                if len(sequential_output) == self.Q:
                    break
        
        sequential_output = tf.concat(sequential_output, 1)
        # sequential_output.shape (batch_size , Q,  data_height , data_width, 1)
        
        return sequential_output
