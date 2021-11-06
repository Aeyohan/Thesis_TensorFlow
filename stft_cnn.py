import param
from stft import *
import tensorflow as tf
import os
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

from tensorflow.python.ops.gen_batch_ops import batch
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import logging
tf.get_logger().setLevel(logging.ERROR)
import tensorflow_hub as hub
import tensorflow_io as tfio
tf.config.list_physical_devices()
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
# %% function to grab the stft of a loaded wav file
def get_stft(wav_data):
    data = wav_data
    if wav_data.shape[0] > 10500:
        subset = tf.slice(wav_data, begin=[1], size=[10501])
        data = subset
    if wav_data.shape[0] < 10500:
        padding = tf.constant([[0, 10500 - wav_data.shape[0]]])
        subset = tf.pad(data,padding, "CONSTANT")
        data = subset
    spectrum = waveform_to_log_mel_spectrogram(data, param)
    return spectrum


def get_batch_stft(data):
    return [get_stft(sample) for sample in data]
    
def load_txt_mono_raw(path):
    file = open(path, 'r')
    # read the first line of no-value
    file.readline()
    values = [line for line in file]
    file.close()
    values = [1 if value > 0.99 else (-1 if value < -0.99 else value) for value in values]
    return tf.convert_to_tensor(values)
# %% \

# merge_model = models.Sequential([
#     layers.Input(shape=(num_windows, num_labels)),
#     layers.Flatten(),
#     layers.Dense(64),
#     layers.Dense(len(num_labels))
# ], name='merge_model')
# %%
