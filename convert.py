# %%
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

import stft_cnn

existing_model = tf.saved_model.load("models/stft_u") 
# existing_small_model = tf.saved_model.load("./models/spec_g_16-32-32_38-84")
existing_small_model = tf.saved_model.load("./models/spec_g_12-16-16_49-70")
class STFTOperation(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[1050], dtype=tf.float32)])
    def __call__(self,x):
        return waveform_to_log_mel_spectrogram(x, param)
        # have to add on the aditional slice?
        # stage1 = tf.expand_dims(stage1, 2)
        # stage1 = tf.expand_dims(stage1, 0)
        # return existing_model(stage1)

class CNNOp(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[32,32], dtype=tf.float32)])
    def __call__(self,x):
        s1 = tf.expand_dims(x,2)
        s1 = tf.expand_dims(s1,0)
        return existing_small_model(s1)

# %%
if __name__ == "__main__":
    model = CNNOp()
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
    converter.target_spec.supported_types = [tf.int16, tf.uint8, tf.uint32, tf.int32, tf.int8, tf.float16] 
    tflite_model = converter.convert()
    with open('lite_models/compound_model_test.tflite', 'wb') as f:
        f.write(tflite_model)
    pass

    
# %%


