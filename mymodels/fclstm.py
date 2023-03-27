import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, ConvLSTM2D, LSTM, RepeatVector, TimeDistributed
from utils.loss import *
from submodules.pos_encoding_2d import *

class MyFCLSTM(tf.keras.models.Model):
    def __init__(self, args, metadata):
        super(MyFCLSTM, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.D = args.D
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyFCLSTM'
                
    def build(self, input_shape):
        self.encoder = LSTM(self.D, return_sequences=True, return_state=True, name='encoder')
        self.decoder = LSTM(self.D, return_sequences=True, return_state=True, name='decoder')
        
        self.embed_layer = keras.models.Sequential([
                                layers.Dense(self.D, activation='leaky_relu'),
                                layers.Dense(self.D, activation='leaky_relu'),
                                layers.Dense(self.D),
                                layers.BatchNormalization()])
        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, activation='leaky_relu'),
                                layers.Dense(self.D, activation='leaky_relu'),
                                layers.Dense(self.num_cells)])


    def call(self, X, TE):
        assert TE.shape[1] == self.P + self.Q
        batch_size = tf.shape(X)[0]
        X0 = X
        last_value = X0[:, -1:, ...]

        TE = tf.cast(tf.one_hot(TE[..., 0]*24 + TE[..., 1], depth=24*7), dtype=tf.float32)
        X = tf.reshape(X, (batch_size, self.P, self.num_cells))
        X = tf.concat((X, TE[:, :self.P, ...]), -1)
        X = self.embed_layer(X)
        encoder_outputs, state_h, state_c = self.encoder(X)
        
        sequential_output = []
        start_token = tf.zeros((batch_size, 1, self.D))
        dstate_h, dstate_c= state_h, state_c
        for j in range(self.Q):
            output_value, dstate_h, dstate_c = self.decoder(start_token, initial_state=(dstate_h, dstate_c))
            
            sequential_output.append(output_value)
            if len(sequential_output) == self.Q:
                break
            
            start_token = output_value
        
        sequential_output = tf.stack(sequential_output, 1)
        sequential_output = tf.squeeze(sequential_output, 2)
        sequential_output = self.output_layer(sequential_output)
        sequential_output = tf.reshape(sequential_output, (batch_size, self.Q, self.height, self.width, 1))
        
        sequential_output = last_value + tf.cumsum(sequential_output, axis=1)
        return sequential_output
