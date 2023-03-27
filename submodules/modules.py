import numpy as np
import tensorflow as tf
from tensorflow import keras
from submodules.pos_encoding_2d import *
        
class STEmbedding(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super(STEmbedding, self).__init__()
        self.height = metadata['height']
        self.width = metadata['width']
        self.D = args.D
        self.SE = get_area_encoding()

    def build(self, input_shape):
        self.FC_TE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        self.FC_SE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self):
        
        

        TE_components = [
            tf.one_hot(TE[..., 0], depth = 7), 
            tf.one_hot(TE[..., 1], depth = 24)
        ]
        TE = tf.concat(TE_components)
        TE = tf.expand_dims(TE, axis = 2)
        TE = self.FC_TE(TE)
        
        STE = SE + TE
        
        return STE_C, STE_P, STE_Q
