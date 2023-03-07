import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.loss import *

class MyRWRModel(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(MyRWRModel, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.height*self.width
        self.k_step = 12
        self.teacher = args.teacher
        self.model_name = f'MyRWRModel'
        
        # make conv2d kernels
        kernels = []
        arr = np.arange(9).reshape(3, 3)
        for i in [0, 1, 2, 3,  5, 6, 7, 8]:
            kernels.append(np.array(arr == i).astype(np.float32))
        self.kernels = np.expand_dims(np.expand_dims(np.stack(kernels, 0), -1), -1)
        # self.kernels: (8, 3, 3, 1, 1)
        
        pos_encoding = []
        for j in range(self.height):
            for i in range(self.width):
                #one_hot_height = tf.one_hot(j, depth=height)
                #one_hot_width = tf.one_hot(i, depth=width)
                #pos_encoding.append(tf.concat((one_hot_height, one_hot_width), -1))
                one_hot = tf.one_hot(j*self.width + i, depth=self.num_cells)
                pos_encoding.append(one_hot)
        pos_encoding = tf.stack(pos_encoding, 0)
        #self.pos_encoding = tf.cast(tf.reshape(pos_encoding, (height, width, height+width)), dtype=tf.float32)
        self.pos_encoding = tf.cast(tf.reshape(pos_encoding, (self.height, self.width, self.num_cells)), dtype=tf.float32)
        
        
    def get_config(self):
        config = super().get_config()
        config.update({
        })
        return config

    def build(self, input_shape):
        self.restart_dense = layers.Dense(1, use_bias=False)
        self.bias_dense = layers.Dense(1, use_bias=False)
        self.trans_dense = layers.Dense(8, use_bias=False)

        
        sampling_TE = []
        for i in range(24*7):
            te_i = tf.cast(tf.one_hot(i, depth=24*7), dtype=tf.float32)
            sampling_TE.append(te_i)
        sampling_TE = tf.stack(sampling_TE, 0)
        sampling_TE = tf.expand_dims(tf.expand_dims(sampling_TE, 1), 1)
        sampling_TE = tf.tile(sampling_TE, [1, self.height, self.width, 1])
        sampling_pos = self.pos_encoding
        sampling_pos = tf.expand_dims(sampling_pos, 0)
        sampling_pos = tf.tile(sampling_pos, [24*7, 1, 1, 1])
        self.sampling_TE_pos = tf.concat((sampling_TE, sampling_pos), -1)

        
    def tfb_custom_summary(self):
        trans_prob = self.sampling_TE_pos @ self.trans_dense.kernel
        restart_prob = self.sampling_TE_pos @ self.restart_dense.kernel
        bias_prob = self.sampling_TE_pos @ self.bias_dense.kernel

        return trans_prob, restart_prob, bias_prob

    
    def call(self, X, TE):
        # X.shape (batch_size, P, data_height, data_width, 1)
        # TE.shape (batch_size, P+Q, 2)
        X_original = X
        batch_size = tf.shape(X)[0]
        
        TE = tf.cast(tf.one_hot(TE[..., 0]*24 + TE[..., 1], depth=24*7), dtype=tf.float32)
        # TE.shape (batch_size, P+Q, 7+24)
        
        TE = tf.expand_dims(tf.expand_dims(TE, 2), 2)
        TE = tf.tile(TE, [1, 1, self.height, self.width, 1])
        # TE.shape (batch_size, P+Q, data_height, data_width, 31)
        
        pos_encoding = self.pos_encoding
        pos_encoding = tf.expand_dims(tf.expand_dims(pos_encoding, 0), 0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, self.P+self.Q, 1, 1, 1])
        TE_pos = tf.concat((TE, pos_encoding), -1)
        # TE_pos.shape (batch_size, P+Q, data_height, data_width, 31+height+width)
        TE_pos_P, TE_pos_Q = TE_pos[:, :self.P, :], TE_pos[:, self.P:, :]
        
        
        trans_prob = tf.nn.softmax(self.trans_dense(TE_pos_P), -1)
        trans_prob = tf.expand_dims(trans_prob, -1)
        # trans_prob.shape (batch_size, P, data_height, data_width, 8, 1)
        
        
        restart_prob = tf.nn.sigmoid(self.restart_dense(TE_pos_P))
        X_bias = self.bias_dense(TE_pos_P) * X
        #self.restart # 0.15



        X_results = [X]
        for k in range(self.k_step):
            X0 = X
            # X0.shape (batch_size, num_seq, data_height, data_width, 1)

            Xn = tf.expand_dims(X, -1) * trans_prob
            # Xn.shape (batch_size, P, data_height, data_width, 8, 1)
            Xn = tf.transpose(Xn, (0, 1, 4, 2, 3, 5))
            # Xn.shape (batch_size, P, 8, data_height, data_width, 1)
            Xn = tf.pad(Xn, [[0,0],[0,0],[0,0],[1,1],[1,1],[0,0]])
            # Xn.shape (batch_size, P, 8, data_height+2, data_width+2, 1)
            X_move = tf.reshape(Xn, (-1, 8, self.height+2, self.width+2, 1))
            X_trans = tf.nn.conv3d(X_move, self.kernels, strides=[1, 1, 1, 1, 1], padding='VALID')
            X_trans = tf.reshape(X_trans, (-1, self.P, 1, self.height, self.width, 1))
            # X_trans.shape (batch_size, P, 1, data_height, data_width, 1)
            X_trans = tf.squeeze(X_trans, 2)
            # X_trans.shape (batch_size, P, data_height, data_width, 1)
            X = (1-restart_prob) * X_trans + restart_prob * X0 + X_bias
            X_results.append(X)
            
        
        Y = X[:, -1:, ...]
        
        if self.teacher > 0:
            loss = self.teacher * custom_mae_loss(X_original[:, 1:, ...], X[:, :-1, ...])
            self.add_loss(loss)
        
        return Y, X, trans_prob, restart_prob, tf.stack(X_results, -1)
    