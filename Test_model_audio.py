# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:00:17 2021

@author: SHIRISH
"""


import Model as m
import tensorflow as tf
import os
import numpy as np
import Utility as u




sampling_rate = 16000

model = m.build_model((sampling_rate//2,1),30)

#model.summary()

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.load_weights('Speech_To_Text_model.h5')

class_names_dict ={}
i = 0
for files in os.listdir('augmented_dataset'):
    class_names_dict[i]=files
    i+=1


audio_path = []
file_name = os.path.join("Test","9.wav")
audio_path.append(file_name)
label =[0]

test_ds = u.GenDataset(audio_path,label)

for audio,_ in test_ds.take(1):
    #add extra dimension to audio
    audio = np.asarray(audio)
    audio = audio.reshape((1,)+audio.shape)
    audio = tf.convert_to_tensor(audio)
    ffts = u.Fourier_transform(audio)
    prediction = model.predict(ffts)
    print("The sound predicted is ",class_names_dict[np.argmax(prediction)])
