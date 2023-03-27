import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, ConvLSTM2D, Conv2D, LSTM, RepeatVector, TimeDistributed
from utils.loss import *
from submodules.pos_encoding_2d import *

class MyConvLSTM(tf.keras.models.Model):
    def __init__(self, args, metadata):
        super(MyConvLSTM, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.D = args.D
        self.kernel_size = (args.conv_kernel_size, args.conv_kernel_size)
        self.teacher = args.teacher
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyConvLSTM_{self.kernel_size}'
        
        # self.pos_encoding = get_ae_pos_encoding()
        self.pos_encoding = get_2d_onehot_encoding(self.height, self.width)
        # self.pos_encoding = get_2d_pos_encoding(self.height, self.width, self.D)
        
        
    def build(self, input_shape):
        self.norm_w = self.add_weight(shape=(1, 1, self.height, self.width, 1), initializer="random_normal", trainable=True, name='norm_w')
        self.norm_b = self.add_weight(shape=(1, 1, self.height, self.width, 1), initializer="random_normal", trainable=True, name='norm_b')
        # if self.K > 1:
        #     self.encoder_stack = keras.models.Sequential([
        #         ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, padding='same') for _ in range(self.K-1)
        #     ])
        #     self.decoder_stack = keras.models.Sequential([
        #         ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, padding='same') for _ in range(self.K-1)
        #     ])
        self.encoder = ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, return_state=True, padding='same')
        self.decoder = ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, return_state=True, padding='same')
        
        self.embed_layer = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='leaky_relu'),
                                layers.Dense(self.D, use_bias=False)])
        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='leaky_relu'),
                                layers.Dense(self.D, use_bias=False, activation='leaky_relu'),
                                layers.Dense(1, use_bias=False)])
        

    def call(self, X, TE):
        # X.shape (batch_size, P, data_height, data_width, 1)
        # TE.shape (batch_size, P+Q, 2)
        assert TE.shape[1] == self.P + self.Q
        batch_size = tf.shape(X)[0]
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
        
        # embX = tf.concat((X, TE_pos_P), -1)
        embX = X
        # embX.shape  (batch_size, P, data_height, data_width, 1+24*7+height*width)
        embX = self.embed_layer(embX)
        # embX.shape  (batch_size, P, data_height, data_width, D)
        # if self.K > 1:
        #     embX = self.encoder_stack(embX)
        encoder_outputs, state_h, state_c = self.encoder(embX)
        # encoder_outputs.shape  (batch_size, P, data_height, data_width, D)
        
        if self.teacher > 0:
            self_outputs = self.output_layer(encoder_outputs) # hidden_size -> 1
            # self_outputs.shape  (batch_size, P, data_height, data_width, 1)
            loss = self.teacher * custom_mae_loss(X[:, 1:, ...], self_outputs[:, :-1, ...])
            self.add_weight(loss)
            
        
        sequential_output = []
        #start_token = tf.zeros((batch_size, 1, self.height, self.width, self.D))
        start_token = embX[:, -1:, :]
        dstate_h, dstate_c= state_h, state_c
        for j in range(self.Q):
            # if self.K > 1:
            #     start_token = self.decoder_stack(start_token)
            decoder_outputs, dstate_h, dstate_c = self.decoder(start_token, initial_state=(dstate_h, dstate_c))
            
            output_value = self.output_layer(decoder_outputs)
            sequential_output.append(output_value)
            if len(sequential_output) == self.Q:
                break
            
            TEQj_enc = tf.expand_dims(TE_pos_Q[:, j, ...], 1)
            next_X = tf.concat((output_value, TEQj_enc), -1)
            start_token = self.embed_layer(next_X)
        
        sequential_output = tf.concat(sequential_output, 1)
        # sequential_output.shape (batch_size , Q,  data_height , data_width, 1)
        
        sequential_output = sequential_output * self.norm_w + self.norm_b
        return sequential_output


class MyConvLSTMD(tf.keras.models.Model):
    def __init__(self, args, metadata):
        super(MyConvLSTMD, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.D = args.D
        self.kernel_size = (args.conv_kernel_size, args.conv_kernel_size)
        self.teacher = args.teacher
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyConvLSTMD_{self.kernel_size}'
        
        self.pos_encoding = get_2d_onehot_encoding(self.height, self.width)

        
        # self.pos_encoding = get_ae_pos_encoding()
        # self.pos_encoding = get_2d_onehot_encoding(self.height, self.width)
        # self.pos_encoding = get_2d_pos_encoding(self.height, self.width, self.D)
        
        
    def build(self, input_shape):
        self.norm_w = self.add_weight(shape=(1, 1, self.height, self.width, 1), initializer="random_normal", trainable=True, name='norm_w')
        self.norm_b = self.add_weight(shape=(1, 1, self.height, self.width, 1), initializer="random_normal", trainable=True, name='norm_b')
        # if self.K > 1:
        #     self.encoder_stack = keras.models.Sequential([
        #         ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, padding='same') for _ in range(self.K-1)
        #     ])
        #     self.decoder_stack = keras.models.Sequential([
        #         ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, padding='same') for _ in range(self.K-1)
        #     ])
        self.encoder = ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, return_state=True, padding='same')
        self.decoder = ConvLSTM2D(self.D, self.kernel_size, return_sequences=True, return_state=True, padding='same')
        
        # self.embed_layer = layers.Dense(self.D, use_bias=False)
        
        self.embed_layer = Conv2D(filters=self.D, kernel_size=(1, 1), activation='relu')
        
        self.output_layer = keras.models.Sequential([
                                Conv2D(filters=self.D, kernel_size=(1, 1), activation='relu'),
                                Conv2D(filters=1, kernel_size=(1, 1))])
        # self.output_layer = keras.models.Sequential([
        #                         layers.Dense(self.D, use_bias=False, activation='leaky_relu'),
        #                         layers.Dense(self.D, use_bias=False, activation='leaky_relu'),
        #                         layers.Dense(1, use_bias=False)])
        

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
        # TE_pos.shape (batch_size, P+Q, data_height, data_width, 24*7+height*width)
        TE_pos_P, TE_pos_Q = TE_pos[:, :self.P-1, ...], TE_pos[:, self.P-1:, ...]
        
        embX = tf.concat((X_diff, TE_pos_P), -1)
        # embX = X_diff
        # embX.shape  (batch_size, P, data_height, data_width, 1+24*7+height*width)
        embX = self.embed_layer(embX)
        # embX.shape  (batch_size, P, data_height, data_width, D)
        # if self.K > 1:
        #     embX = self.encoder_stack(embX)
        encoder_outputs, state_h, state_c = self.encoder(embX)
        # encoder_outputs.shape  (batch_size, P, data_height, data_width, D)
        
        # if self.teacher > 0:
        #     self_outputs = self.output_layer(encoder_outputs) # hidden_size -> 1
        #     # self_outputs.shape  (batch_size, P, data_height, data_width, 1)
        #     loss = self.teacher * custom_mae_loss(X[:, 1:, ...], self_outputs[:, :-1, ...])
        #     self.add_weight(loss)
            
        
        last_value = X[:, -1:, ...]
        sequential_output = []
        start_token = tf.zeros((batch_size, 1, self.height, self.width, self.D))
        # start_token = embX[:, -1:, :]
        dstate_h, dstate_c= state_h, state_c
        for j in range(self.Q):
            # if self.K > 1:
            #     start_token = self.decoder_stack(start_token)
            decoder_outputs, dstate_h, dstate_c = self.decoder(start_token, initial_state=(dstate_h, dstate_c))
            
            output_value = self.output_layer(decoder_outputs)
            output_value_s = tf.reshape(output_value, (-1, 1, self.height, self.width, 1))
            last_value = last_value + output_value_s
            sequential_output.append(last_value)
            if len(sequential_output) == self.Q:
                break


            # output_value = self.output_layer(decoder_outputs)
            # sequential_output.append(output_value)
            # if len(sequential_output) == self.Q:
            #     break
            
            TEQj_enc = tf.expand_dims(TE_pos_Q[:, j, ...], 1)
            next_X = tf.concat((output_value, TEQj_enc), -1)
            start_token = self.embed_layer(next_X)
        
        sequential_output = tf.concat(sequential_output, 1)
        # sequential_output.shape (batch_size , Q,  data_height , data_width, 1)
        
        sequential_output = sequential_output * self.norm_w + self.norm_b
        return sequential_output
