import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from submodules.pos_encoding_2d import *


class STEmbedding(tf.keras.layers.Layer):
    def __init__(self, D):
        super(STEmbedding, self).__init__()
        self.D = D

    def build(self, input_shape):
        self.FC_TE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        self.FC_SE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, SE, TE):
        # dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        # timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # TE = tf.concat((dayofweek, timeofday), axis = -1)
        # TE = tf.expand_dims(TE, axis = 2)
        TE = self.FC_TE(TE)
        SE = self.FC_SE(SE)
        
        STE = SE + TE
        return STE


class MyTemporalTransformer(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(MyTemporalTransformer, self).__init__()
        self.P = args.P
        self.Q = args.Q
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.teacher = args.teacher
        self.height = metadata['data_height']
        self.width = metadata['data_width']
        self.num_cells = self.width * self.height
        self.model_name = f'MyTemporalTransformer'
        self.pos_encoding = get_2d_pos_encoding(self.height, self.width, self.D)
        
    def build(self, input_shape):
        # self.norm_w = self.add_weight(shape=(1, 1, self.height, self.width, 1), initializer="random_normal", trainable=True, name='norm_w')
        self.embed_layer = keras.models.Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(self.D),
                                layers.BatchNormalization()])
        self.output_layer = keras.models.Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(1)])
            
        self.STE_layer = STEmbedding(self.D)
        
        self.ST_encoder = [TemporalAttention(self.K, self.d) for _ in range(self.L)]
        self.trans_layer = TransformAttention(self.K, self.d)
        self.ST_decoder = [TemporalAttention(self.K, self.d) for _ in range(self.L)]
        
        
    def call(self, X, TE):
        # X.shape (batch_size, P, data_height, data_width, 1)
        # TE.shape (batch_size, P+Q, 2)
        assert TE.shape[1] == self.P + self.Q
        batch_size = tf.shape(X)[0]
        X0 = X
        last_value = X0[:, -1:, ...]

        
        TE = tf.cast(tf.concat((tf.one_hot(TE[..., 0], depth=7), tf.one_hot(TE[..., 1], depth=24)), -1), dtype=tf.float32)
        # TE = tf.cast(tf.one_hot(TE[..., 0]*24 + TE[..., 1], depth=24*7), dtype=tf.float32)
        TE = tf.expand_dims(tf.expand_dims(TE, 2), 2)
        TE = tf.tile(TE, [1, 1, self.height, self.width, 1])
        pos_encoding = self.pos_encoding
        pos_encoding = tf.expand_dims(tf.expand_dims(pos_encoding, 0), 0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, self.P+self.Q, 1, 1, 1])

        STE = self.STE_layer(pos_encoding, TE)
        X = self.embed_layer(X)

        X = tf.reshape(X, (-1, self.P, self.num_cells, self.D))
        STE = tf.reshape(STE, (-1, self.P+self.Q, self.num_cells, self.D))
        STE_P, STE_Q = STE[:, :self.P, ...], STE[:, self.P:, ...]
        
        for i in range(self.L):
            X = self.ST_encoder[i](X, STE_P)
        X = self.trans_layer(X, STE_P, STE_Q)
        for i in range(self.L):
            X = self.ST_decoder[i](X, STE_Q)
        X = self.output_layer(X)
        Y = tf.reshape(X, (-1, self.Q, self.height, self.width, 1))

        # Y = last_value + tf.cumsum(Y, axis=1)

        return Y




class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(SpatialAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.BatchNormalization()])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.BatchNormalization()])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.BatchNormalization()])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),
            layers.BatchNormalization()])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        attention = tf.matmul(query, key, transpose_b = True)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, K, d, use_mask=False):
        super(TemporalAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d
        self.use_mask = use_mask

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.BatchNormalization()])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.BatchNormalization()])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.BatchNormalization()])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),
            layers.BatchNormalization()])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        if self.use_mask:
            batch_size = tf.shape(X)[0]
            num_step = tf.shape(X)[1]
            N = tf.shape(X)[2]
            mask = tf.ones(shape = (num_step, num_step))
            mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
            mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
            mask = tf.tile(mask, multiples = (K * batch_size, N, 1, 1))
            mask = tf.cast(mask, dtype = tf.bool)
            attention = tf.compat.v2.where(
                condition = mask, x = attention, y = -2 ** 15 + 1)
            
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
    
class GatedFusion(tf.keras.layers.Layer):
    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.D = D

    def build(self, input_shape):
        self.FC_S = keras.Sequential([
            layers.Dense(self.D, use_bias=False),
            layers.BatchNormalization()])
        self.FC_T = keras.Sequential([
            layers.Dense(self.D),
            layers.BatchNormalization()])
        self.FC_H = keras.Sequential([
            layers.Dense(self.D, activation='relu'),
            layers.Dense(self.D),
            layers.BatchNormalization()])
        
    def call(self, HS, HT):
        XS = self.FC_S(HS)
        XT = self.FC_T(HT)
        
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = self.FC_H(H)
        return H
    
class STAttBlock(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(STAttBlock, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.SA_layer = SpatialAttention(self.K, self.d)
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        
        HS = self.SA_layer(X, STE)
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        return X + H
    

class TransformAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(TransformAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.BatchNormalization()])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.BatchNormalization()])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.BatchNormalization()])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),
            layers.BatchNormalization()])
        
    def call(self, X, STE_P, STE_Q):
        K = self.K
        d = self.d
        D = self.D
        
        query = self.FC_Q(STE_Q)
        key = self.FC_K(STE_P)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))   
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X