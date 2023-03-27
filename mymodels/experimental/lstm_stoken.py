import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from utils.loss import *
from submodules.pos_encoding_2d import *

class MyARLSTMST(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(MyARLSTMST, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.D = args.D
        self.teacher = args.teacher
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyARLSTMST'
        
        
        # self.pos_encoding = get_ae_pos_encoding()
        self.pos_encoding = get_2d_pos_encoding(self.height, self.width, self.D)
        # pos_encoding = []
        # for j in range(self.height):
        #     for i in range(self.width):
        #         one_hot = tf.one_hot(j*self.width + i, depth=self.num_cells)
        #         pos_encoding.append(one_hot)
        # pos_encoding = tf.stack(pos_encoding, 0)
        # self.pos_encoding = tf.cast(tf.reshape(pos_encoding, (self.height, self.width, self.num_cells)), dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "P": self.P,
            "Q": self.Q,
            "K": self.K,
            "D": self.D,
            "teacher": self.teacher,
        })
        return config
        
    def build(self, input_shape):
        self.norm_w = self.add_weight(shape=(1, 1, self.height, self.width, 1), initializer="random_normal", trainable=True, name='norm_w')
        self.norm_b = self.add_weight(shape=(1, 1, self.height, self.width, 1), initializer="random_normal", trainable=True, name='norm_b')
        self.encoder = LSTM(self.D, return_sequences=True, return_state=True, name='encoder')
        self.decoder = LSTM(self.D, return_sequences=True, return_state=True, name='decoder')
        
        self.embed_layer = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='relu'),
                                layers.Dense(self.D, use_bias=False)])
        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='relu'),
                                layers.Dense(1, use_bias=False)])

        self.config = self.get_config()
    
    def call(self, X, TE):
        # X.shape (batch_size, P, data_height, data_width, 1)
        # TE.shape (batch_size, P+Q, 2)
        assert TE.shape[1] == self.P + self.Q
        batch_size = tf.shape(X)[0]

        # X_mean = tf.expand_dims(tf.math.reduce_mean(X, axis=1), 1)
        # X = X - X_mean

        X = (X - self.norm_b) / self.norm_w
        
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
        
        sequential_output = sequential_output * self.norm_w + self.norm_b
        return sequential_output




class MyARLSTMSTD(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(MyARLSTMSTD, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.D = args.D
        self.teacher = args.teacher
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyARLSTMSTD'
        
        # self.pos_encoding = get_area_encoding()
        # self.pos_encoding = get_ae_pos_encoding()
        self.pos_encoding = get_2d_onehot_encoding(self.height, self.width)
        # self.pos_encoding = get_2d_pos_encoding(self.height, self.width, self.D)

    def get_config(self):
        config = super().get_config()
        config.update({
            "P": self.P,
            "Q": self.Q,
            "K": self.K,
            "D": self.D,
            "teacher": self.teacher,
        })
        return config
        
    def build(self, input_shape):
        self.norm_w = self.add_weight(shape=(1, 1, self.height, self.width, 1), initializer="random_normal", trainable=True, name='norm_w')
        self.norm_b = self.add_weight(shape=(1, 1, self.height, self.width, 1), initializer="random_normal", trainable=True, name='norm_b')
        self.encoder = LSTM(self.D, return_sequences=True, return_state=True, name='encoder')
        self.decoder = LSTM(self.D, return_sequences=True, return_state=True, name='decoder')
        
        self.embed_layer = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='relu'),
                                layers.Dense(self.D, use_bias=False)])
        self.embed_layer2 = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='relu'),
                                layers.Dense(self.D, use_bias=False)])
        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='relu'),
                                layers.Dense(self.D, use_bias=False, activation='relu'),
                                layers.Dense(1, use_bias=False)])

        self.config = self.get_config()
    
    def call(self, X, TE):
        # X.shape (batch_size, P, data_height, data_width, 1)
        # TE.shape (batch_size, P+Q, 2)
        assert TE.shape[1] == self.P + self.Q
        TE = TE[:, 1:, :]
        batch_size = tf.shape(X)[0]

        X = (X - self.norm_b) / self.norm_w

        X_diff = X[:, 1:, ...] - X[:, :-1, ...]
        
        #TE = tf.cast(tf.concat((tf.one_hot(TE[..., 0], depth=7), tf.one_hot(TE[..., 1], depth=24)), -1), dtype=tf.float32)
        TE = tf.cast(tf.one_hot(TE[..., 0]*24 + TE[..., 1], depth=24*7), dtype=tf.float32)
        # TE.shape (batch_size, P+Q, 7+24)
        
        TE = tf.expand_dims(tf.expand_dims(TE, 2), 2)
        TE = tf.tile(TE, [1, 1, self.height, self.width, 1])
        # TE.shape (batch_size, P+Q, data_height, data_width, 31)
        
        pos_encoding = self.pos_encoding
        pos_encoding = tf.expand_dims(tf.expand_dims(pos_encoding, 0), 0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, self.P+self.Q-1, 1, 1, 1])
        TE_pos = tf.concat((TE, pos_encoding), -1)
        # TE_pos = tf.concat((pos_encoding), -1)
        # TE_pos = tf.concat((TE), -1)
        # TE_pos.shape (batch_size, P+Q, data_height, data_width, 24*7+height*width)
        TE_pos_P, TE_pos_Q = TE_pos[:, :self.P-1, ...], TE_pos[:, self.P-1:, ...]
        
        # embX = tf.concat((X_diff, TE_pos_P), -1)
        embX = X_diff
        # embX.shape  (batch_size, P, data_height, data_width, 1+24*7+height*width)
        embX = self.embed_layer(embX)
        embX = tf.transpose(embX, (0, 2, 3, 1, 4))
        # embX.shape  (batch_size, data_height, data_width, P, D)
        embX = tf.reshape(embX, (-1, self.P-1, self.D))
        # X.shape  (batch_size * data_height * data_width, P, D)
        
        encoder_outputs, state_h, state_c = self.encoder(embX)
        
        # if self.teacher > 0:
        #     self_outputs = self.output_layer(encoder_outputs) # hidden_size -> 1
        #     self_outputs = tf.reshape(self_outputs, (batch_size, self.height, self.width, self.P, 1))
        #     self_outputs = tf.transpose(self_outputs, (0, 3, 1, 2, 4))
        #     loss = self.teacher * custom_mae_loss(X[:, 1:, ...], self_outputs[:, :-1, ...])
        #     self.add_loss(loss)
            
        
        last_value = X[:, -1:, ...]
        sequential_output = []
        # start_token = tf.zeros((tf.shape(embX)[0], 1, self.D)) # start_token = embX[:, -1:, :]
        start_token = self.embed_layer2(TE_pos_Q)[:, :1, ...]
        start_token = tf.transpose(start_token, (0, 2, 3, 1, 4))
        start_token = tf.reshape(start_token, (-1, 1, self.D))
        dstate_h, dstate_c= state_h, state_c
        for j in range(self.Q):
            decoder_outputs, dstate_h, dstate_c = self.decoder(start_token, initial_state=(dstate_h, dstate_c))
            
            output_value = self.output_layer(decoder_outputs)
            output_value_s = tf.reshape(output_value, (-1, 1, self.height, self.width, 1))
            last_value = last_value + output_value_s
            sequential_output.append(last_value)
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
        # sequential_output = tf.reshape(sequential_output, (-1, self.height, self.width, self.Q, 1))
        # sequential_output = tf.transpose(sequential_output, (0, 3, 1, 2, 4))
        
        sequential_output = sequential_output * self.norm_w + self.norm_b
        return sequential_output