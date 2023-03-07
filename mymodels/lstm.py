import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from utils.loss import *


class MyARLSTM(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(MyARLSTM, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.D = args.D
        self.teacher = args.teacher
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyARLSTM'
        
        
        pos_encoding = []
        for j in range(self.height):
            for i in range(self.width):
                one_hot = tf.one_hot(j*self.width + i, depth=self.num_cells)
                pos_encoding.append(one_hot)
        pos_encoding = tf.stack(pos_encoding, 0)
        self.pos_encoding = tf.cast(tf.reshape(pos_encoding, (self.height, self.width, self.num_cells)), dtype=tf.float32)
        
        
    def build(self, input_shape):
        self.encoder = LSTM(self.D, return_sequences=True, return_state=True)
        self.decoder = LSTM(self.D, return_sequences=True, return_state=True)
        
        self.embed_layer = layers.Dense(self.D, use_bias=False)
        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='relu'),
                                layers.Dense(self.D, use_bias=False, activation='relu'),
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
        embX = tf.transpose(embX, (0, 2, 3, 1, 4))
        # embX.shape  (batch_size, data_height, data_width, P, D)
        embX = tf.reshape(embX, (-1, self.P, self.D))
        # X.shape  (batch_size * data_height * data_width, P, D)
        
        encoder_outputs, state_h, state_c = self.encoder(embX)
        
        if self.teacher > 0:
            self_outputs = self.output_layer(encoder_outputs) # hidden_size -> 1
            self_outputs = tf.reshape(self_outputs, (batch_size, self.height, self.width, self.P, 1))
            self_outputs = tf.transpose(self_outputs, (0, 3, 1, 2, 4))
            loss = self.teacher * custom_mae_loss(X[:, 1:, ...], self_outputs[:, :-1, ...])
            self.add_loss(loss)
            
        
        sequential_output = []
        start_token = tf.zeros((tf.shape(embX)[0], 1, self.D)) # start_token = embX[:, -1:, :]
        dstate_h, dstate_c= state_h, state_c
        for j in range(self.Q):
            decoder_outputs, dstate_h, dstate_c = self.decoder(start_token, initial_state=(dstate_h, dstate_c))
            
            output_value = self.output_layer(decoder_outputs)
            sequential_output.append(output_value)
            if len(sequential_output) == self.Q:
                break
            
            TEQj_enc = TE_pos_Q[:, j, ...]
            # TEQj_enc.shape (1, 54, 67, 3786)
            TEQj_enc = tf.reshape(TEQj_enc, (-1, 1, TEQj_enc.shape[-1]))
            # TEQj_enc.shape (54*67, 1, 3786)
            
            next_X = tf.concat((output_value, TEQj_enc), -1)
            start_token = self.embed_layer(next_X)
        
        sequential_output = tf.concat(sequential_output, 1)
        # sequential_output.shape (batch_size * data_height * data_width, Q, 1)
        sequential_output = tf.reshape(sequential_output, (-1, self.height, self.width, self.Q, 1))
        sequential_output = tf.transpose(sequential_output, (0, 3, 1, 2, 4))
        
        return sequential_output