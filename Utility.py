# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:23:17 2021

@author: SHIRISH
"""


import tensorflow as tf
import numpy as np



sampling_rate = 16000
def getAudio(file_path):
    file = tf.io.read_file(file_path)
    
    audio,_ = tf.audio.decode_wav(file,1,sampling_rate)
    return audio

def GenDataset(paths,labels):
    
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    audio_ds = paths_ds.map(lambda x: getAudio(x))
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    return tf.data.Dataset.zip((audio_ds,labels_ds))


def Fourier_transform(audio):
    
    audio = tf.squeeze(audio,axis=-1)
    
    ftt = tf.signal.fft(tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)),tf.complex64))
    
    ftt = tf.expand_dims(ftt,axis=-1)
    
    return tf.math.abs(ftt[:,:(audio.shape[1] // 2),:])