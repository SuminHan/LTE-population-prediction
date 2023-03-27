import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, LSTM, GRU, RepeatVector, TimeDistributed, Dense
from utils.loss import *
from submodules.pos_encoding_2d import *

class Seq2SeqLSTM(tf.keras.Model):
    def __init__(self, P, Q, n_features, lstm_units=64, name='name'):
        super(Seq2SeqLSTM, self).__init__(name=name)
        self.encoder = GRU(lstm_units, input_shape=(P, n_features))
        self.repeat_vector = RepeatVector(Q)
        self.decoder = GRU(lstm_units, return_sequences=True)
        self.time_distributed = TimeDistributed(Dense(n_features))

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.repeat_vector(x)
        x = self.decoder(x)
        return self.time_distributed(x)

class MyRAWLSTM(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(MyRAWLSTM, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.D = args.D
        self.teacher = args.teacher
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyRAWLSTM'

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
        self.each_lstm = [Seq2SeqLSTM(self.P, self.Q, 1, lstm_units=self.D, name=f'{n}_lstm') for n in range(self.num_cells)]
    
    def call(self, X, TE):
        # X.shape (batch_size, P, data_height, data_width, 1)
        # TE.shape (batch_size, P+Q, 2)
        assert TE.shape[1] == self.P + self.Q
        batch_size = tf.shape(X)[0]

        # X_mean = tf.expand_dims(tf.math.reduce_mean(X, axis=1), 1)
        # X = X - X_mean

        # X = (X - self.norm_b) / self.norm_w

        outputs = []
        for j in range(self.height):
            for i in range(self.width):
                this_lstm = self.each_lstm[j*self.width + i]
                outputs.append(this_lstm(X[:, :, j, i, :]))

        outputs = tf.concat(outputs, -1)
        outputs = tf.reshape(outputs, (batch_size, self.Q, self.height, self.width, 1))
        return outputs


