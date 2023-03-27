import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from submodules.pos_encoding_2d import *
from utils import *


class LastRepeat(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(LastRepeat, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.model_name = 'LastRepeat'
        
    def call(self, X, TE):
        return tf.tile(X[:, -1:, ...], [1, self.Q, 1, 1, 1])

        
class LastDerivative(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(LastDerivative, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.model_name = 'LastRepeat'
        
    def call(self, X, TE):
        last_diff = X[:, -1:, ...] - X[:, -2:-1, ...]

        X_out = [X[:, -1:, ...] + last_diff]
        for q in range(self.Q):
            if len(X_out) == self.Q:
                break
            last_value = X_out[-1]
            X_out.append(last_value + last_diff)

        X_out = tf.concat(X_out, 1)
        return X_out

        
class BasicDense(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(BasicDense, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.D = args.D
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.model_name = 'BasicDense'
        self.teacher = args.teacher
        self.pos_encoding = get_2d_onehot_encoding(self.height, self.width)
        # self.pos_encoding = get_2d_pos_encoding(self.height, self.width, self.D)
    
    def build(self, input_shape):
        # self.pred_layer = layers.Dense(1, use_bias=False)
        self.pred_layer = keras.models.Sequential([
                                layers.Dense(self.D, activation='leaky_relu'),
                                layers.Dense(self.D, activation='leaky_relu'),
                                layers.Dense(1)])
        
    def call(self, X, TE):
        batch_size = tf.shape(X)[0]

        TE = tf.cast(tf.one_hot(TE[..., 0]*24 + TE[..., 1], depth=24*7), dtype=tf.float32)
        
        # TE_components = [
        #     tf.one_hot(TE[..., 0], depth = 7), 
        #     tf.one_hot(TE[..., 1], depth = 24)
        # ]
        # TE = tf.cast(tf.concat(TE_components, -1), dtype=tf.float32)

        TE = tf.expand_dims(tf.expand_dims(TE, 2), 2)
        TE = tf.tile(TE, [1, 1, self.height, self.width, 1])

        pos_encoding = self.pos_encoding
        pos_encoding = tf.expand_dims(tf.expand_dims(pos_encoding, 0), 0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, self.P+self.Q, 1, 1, 1])
        TE_pos = tf.concat((TE, pos_encoding), -1)
        # TE_pos_P, TE_pos_Q = TE_pos[:, :self.P, ...], TE_pos[:, self.P:, ...]

        last_value = X[:, -1:, ...]
        pred_diff = self.pred_layer(TE_pos)

        # X_diff = X[:, 1:, ...] - X[:, :-1, ...]
        # if self.teacher > 0:
        #     loss = self.teacher * custom_mae_loss(X_diff, pred_diff[:, :self.P-1, ...])
        #     self.add_loss(loss)
        


        sequential_output = []
        for q in range(self.Q):
            output_value = pred_diff[:, self.P+q-1:self.P+q, ...]
            sequential_output.append(output_value)
        sequential_output = tf.concat(sequential_output, 1)
        sequential_output = last_value + tf.cumsum(sequential_output, axis=1)

        return sequential_output
