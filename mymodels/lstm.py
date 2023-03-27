import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from utils.loss import *
from submodules.pos_encoding_2d import *

        
# class MyLSTMDense(tf.keras.layers.Layer):
#     def __init__(self, args, metadata):
#         super(MyLSTMDense, self).__init__()
#         self.P = args.P
#         self.Q = args.Q
#         self.D = args.D
#         self.height = metadata['data_height']
#         self.width = metadata['data_width']
#         self.model_name = 'MyLSTMDense'
#         self.teacher = args.teacher
#         self.pos_encoding = get_2d_onehot_encoding(self.height, self.width)
#         # self.pos_encoding = get_2d_pos_encoding(self.height, self.width, self.D)
    
#     def build(self, input_shape):
#         # self.pred_layer = layers.Dense(1, use_bias=False)
#         self.X_emb_layer = keras.models.Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D)])
#         self.TE_emb_layer = keras.models.Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D)])
#         self.pred_layer = keras.models.Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(1)])
#         self.encoder = LSTM(self.D, name='encoder')
#         self.bn = tf.keras.layers.BatchNormalization()
        
#     def call(self, X, TE):
#         batch_size = tf.shape(X)[0]

#         # TE = tf.cast(tf.one_hot(TE[..., 0]*24 + TE[..., 1], depth=24*7), dtype=tf.float32)
        
#         TE_components = [
#             tf.one_hot(TE[..., 0], depth = 7), 
#             tf.one_hot(TE[..., 1], depth = 24)
#         ]
#         TE = tf.cast(tf.concat(TE_components, -1), dtype=tf.float32)

#         TE = tf.expand_dims(tf.expand_dims(TE, 2), 2)
#         TE = tf.tile(TE, [1, 1, self.height, self.width, 1])

#         pos_encoding = self.pos_encoding
#         pos_encoding = tf.expand_dims(tf.expand_dims(pos_encoding, 0), 0)
#         pos_encoding = tf.tile(pos_encoding, [batch_size, self.P+self.Q, 1, 1, 1])
#         TE_pos = self.TE_emb_layer(tf.concat((TE, pos_encoding), -1))
#         TE_pos_P, TE_pos_Q = TE_pos[:, :self.P, ...], TE_pos[:, self.P:, ...]

#         last_value = X[:, -1:, ...]

#         # X_diff = X[:, 1:, ...] - X[:, :-1, ...]
#         # X_emb = self.X_emb_layer(X_diff) + TE_pos_P[:, 1:, ...]
#         X_emb = self.X_emb_layer(X) + TE_pos_P
#         X_emb = self.bn(X_emb)
#         X_emb = tf.transpose(X_emb, (0, 2, 3, 1, 4))
#         X_emb = tf.reshape(X_emb, (-1, self.P, self.D))
#         X_emb = self.encoder(X_emb)
#         X_emb = tf.reshape(X_emb, (-1, self.height, self.width, 1, self.D))
#         X_emb = tf.transpose(X_emb, (0, 3, 1, 2, 4))

#         pred_diff = self.pred_layer(X_emb)
#         # pred_diff = self.pred_layer(TE_pos_Q)

        
#         # if self.teacher > 0:
#         #     loss = self.teacher * custom_mae_loss(X_diff, pred_diff[:, :self.P-1, ...])
#         #     self.add_loss(loss)
        


#         sequential_output = []
#         for q in range(self.Q):
#             last_value = last_value + pred_diff[:, q:q+1, ...]
#             sequential_output.append(last_value)
#         sequential_output = tf.concat(sequential_output, 1)

#         return sequential_output



class MyARLSTM(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.D = args.D
        self.teacher = args.teacher
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyARLSTM'

        self.pos_encoding = get_2d_pos_encoding(self.height, self.width, self.D)

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
        self.encoder = LSTM(self.D, return_sequences=True, return_state=True, name='encoder')
        self.decoder = LSTM(self.D, return_sequences=True, return_state=True, name='decoder')
        
        self.embed_layer = keras.models.Sequential([
                                layers.Dense(self.D, use_bias=False, activation='relu'),
                                layers.Dense(self.D, use_bias=False, activation='relu'),
                                layers.Dense(self.D),
                                layers.BatchNormalization()])
        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(1)])

        self.config = self.get_config()
    
    def call(self, X, TE):
        # X.shape (batch_size, P, data_height, data_width, 1)
        # TE.shape (batch_size, P+Q, 2)
        assert TE.shape[1] == self.P + self.Q
        batch_size = tf.shape(X)[0]
        X0 = X
        last_value = X0[:, -1:, ...]

        
        # TE = tf.cast(tf.concat((tf.one_hot(TE[..., 0], depth=7), tf.one_hot(TE[..., 1], depth=24)), -1), dtype=tf.float32)
        TE = tf.cast(tf.one_hot(TE[..., 0]*24 + TE[..., 1], depth=24*7), dtype=tf.float32)
        TE = tf.expand_dims(tf.expand_dims(TE, 2), 2)
        TE = tf.tile(TE, [1, 1, self.height, self.width, 1])
        pos_encoding = self.pos_encoding
        pos_encoding = tf.expand_dims(tf.expand_dims(pos_encoding, 0), 0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, self.P+self.Q, 1, 1, 1])
        TE_pos = tf.concat((TE, pos_encoding), -1)
        TE_pos_P, TE_pos_Q = TE_pos[:, :self.P, ...], TE_pos[:, self.P:, ...]
        
        embX = tf.concat((X, TE_pos_P), -1)
        embX = self.embed_layer(embX)
        embX = tf.transpose(embX, (0, 2, 3, 1, 4))
        embX = tf.reshape(embX, (-1, self.P, self.D))
        
        encoder_outputs, state_h, state_c = self.encoder(embX)
        
        sequential_output = []
        start_token = tf.zeros((tf.shape(embX)[0], 1, self.D)) 

        dstate_h, dstate_c= state_h, state_c
        for j in range(self.Q):
            decoder_outputs, dstate_h, dstate_c = self.decoder(start_token, initial_state=(dstate_h, dstate_c))
            
            sequential_output.append(decoder_outputs)
            if len(sequential_output) == self.Q:
                break

            start_token = decoder_outputs
        
        sequential_output = self.output_layer(tf.concat(sequential_output, 1))
        sequential_output = tf.reshape(sequential_output, (-1, self.height, self.width, self.Q, 1))
        sequential_output = tf.transpose(sequential_output, (0, 3, 1, 2, 4))
        
        sequential_output = sequential_output
        # sequential_output = last_value + tf.cumsum(sequential_output, axis=1)

        return sequential_output

