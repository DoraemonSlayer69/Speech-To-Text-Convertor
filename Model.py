# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:37:31 2021

@author: SHIRISH
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras


def Residual_block(x,filters,num_layers=3,activation='relu'):
    
    rc = tf.keras.layers.Conv1D(filters,1,padding='same')(x)
    for i in range(num_layers-1):
        x = tf.keras.layers.Conv1D(filters,3,padding='same')(x)
        x = tf.keras.layers.Activation(activation)(x)
    
    x = tf.keras.layers.Conv1D(filters,3,padding='same')(x)
    x = tf.keras.layers.Add()([x,rc])
    x = tf.keras.layers.Activation(activation)(x)
    return tf.keras.layers.MaxPooling1D(2,2)(x)


def build_model(input_shape,num_classes):
    
    input_layer = tf.keras.layers.Input(shape=input_shape,name='Input')
    x = Residual_block(input_layer,32,2)
    x = Residual_block(x,32,2)
    x = Residual_block(x,64,3)
    x = Residual_block(x,128,3)
    
    x = tf.keras.layers.AveragePooling1D(3,3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64,activation='relu')(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=outputs)