# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 17:40:04 2021

@author: SHIRISH
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
import Utility as u
import Model as network
import os

sampling_rate = 16000


seed = 43

validation_split = 0.25





#Get the dataset
    
root_data = "augmented_dataset"

class_names = os.listdir(root_data)

audio_paths = []

labels = []

for label,name in enumerate(class_names):
    
    temp = os.path.join(root_data,name)
    sample_paths = []
    for files in os.listdir(temp):
        sample_paths.append(os.path.join(temp,files))
    
    audio_paths+=sample_paths
    labels+=[label] * len(sample_paths)


rng = np.random.RandomState(seed)
rng.shuffle(audio_paths)
rng = np.random.RandomState(seed)
rng.shuffle(labels)


num_samples = int(validation_split * len(audio_paths))

train_audio = audio_paths[:-num_samples]
train_labels = labels[:-num_samples]

valid_audio = audio_paths[-num_samples:]
valid_labels = labels[-num_samples:]

tensor_data_train = u.GenDataset(train_audio,train_labels)
tensor_data_val = u.GenDataset(train_audio,train_labels)


tensor_data_train = tensor_data_train.shuffle(128*8,seed=seed).batch(128)
tensor_data_val = tensor_data_val.shuffle(32*8,seed=seed).batch(32)

tensor_data_train = tensor_data_train.map(lambda x,y: (u.Fourier_transform(x),y),num_parallel_calls=tf.data.experimental.AUTOTUNE)

tensor_data_val = tensor_data_val.map(lambda x,y: (u.Fourier_transform(x),y),num_parallel_calls=tf.data.experimental.AUTOTUNE)






model = network.build_model((sampling_rate // 2, 1),len(class_names))

model.summary()    

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',patience=3,verbose=1)

history = model.fit(
    tensor_data_train,
    epochs=10,
    validation_data=tensor_data_val,
    callbacks=es
)

model.save_weights("Speech_To_Text_model.h5")


    