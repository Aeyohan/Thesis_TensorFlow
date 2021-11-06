# %%


import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
import os
import numpy as np

import random


# %%
model_list = [
    # "spec_g_8_12_12/spec_g_8_12_12_4_44-75",
    # "spec_g_8_24_12/spec_g_8_24_12_4_37-82",
    # "spec_g_8_32_32/spec_g_8_32_32_4_33-83",
    # "spec_g_8_48_24/spec_g_8_48_24_4_22-92",
    # "spec_g_16_16_24/spec_g_16_16_24_3_20-93",
    # "spec_g_8_48_80/spec_g_8_48_80_4_26-90",
    # "spec_g_20_24_16/spec_g_20_24_16_4_59-73",
    # "spec_g_8_96_64/spec_g_8_96_64_4_17-94",
    # "spec_g_20_24_32/spec_g_20_24_32_1_29-88",
    # "spec_g_16_32_48/spec_g_16_32_48_3_16-94",
    "spec_g_24_64_80/spec_g_24_64_80_1_8-97",
    # "spec_g_20_48_96/spec_g_20_48_96_4_15-96",
    # "spec_g_16_48_96/spec_g_16_48_96_3_11-96"
]

model_list = [
    "spec_g_16_64_128/spec_g_16_64_128_4_8-98"
]

name_list = [file.split("/")[1] for file in model_list]

path_to = "binary_models"

fl16_save_to="binary_fl16_models"

models = [tf.keras.models.load_model(os.path.join(path_to, file)) for file in model_list]
existing_tflite_model = None

class CNNOp(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[32,32], dtype=tf.int16)])
    def __call__(self,x):
        s1 = tf.expand_dims(x,2)
        s1 = tf.expand_dims(s1,0)
        s1 = tf.cast(s1, dtype=tf.float32)
        return existing_tflite_model(s1)
# %% fl16 models

if not os.path.exists(fl16_save_to):
    os.mkdir(fl16_save_to)

kmodels = [tf.keras.models.load_model(os.path.join(path_to, file)) for file in model_list]
existing_tflite_model = None

for model, model_path in zip(kmodels, model_list):
    existing_tflite_model = model
    hybrid_model = CNNOp()
    concrete_func = hybrid_model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(path_to, model_path))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    destination = os.path.join(fl16_save_to, model_path.split("/")[1] + ".tflite")
    path = "/mnt/c/Users/Aeyohan/Documents/work/ENGG4811/code/Tensorflow/"
    command = 'ubuntu run "xxd -i ' + path + destination + ' > ' + path + destination.split(".tflite")[0]+".cpp" + '"' 
    command = command.replace("\\", "/")
    print(command)
    with open(destination, 'wb') as f:
        f.write(tflite_quant_model)
    os.system(command)
# %% quantisation

fl16_quant_save_to="binary_fl16_quant_models"
# create representative dataset
print("Loading dataset")
rep_ds = load_trimmed_dataset(True) # trainset
print("Finshed loading dataset")
def representative_data_gen():
  for input_value in rep_ds.batch(1).take(500):
    # Model has only one input so each data point has one element.
    yield [input_value]

if not os.path.exists(fl16_quant_save_to):
    os.mkdir(fl16_quant_save_to)


for model, model_path in zip(kmodels, model_list):
    existing_tflite_model = model
    hybrid_model = CNNOp()
    concrete_func = hybrid_model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(path_to, model_path))
    converter.representative_dataset = representative_data_gen
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()
    destination = os.path.join(fl16_quant_save_to, model_path.split("/")[1] + ".tflite")
    with open(destination, 'wb') as f:
        f.write(tflite_quant_model)
    path = "/mnt/c/Users/Aeyohan/Documents/work/ENGG4811/code/Tensorflow/"
    command = 'ubuntu run "xxd -i ' + path + destination + ' > ' + path + destination.split(".tflite")[0]+".cpp" + '"' 
    command = command.replace("\\", "/")
    os.system(command)
        

# %%


int_quant_save_to="binary_int_quant_models"
# create representative dataset
print("Loading dataset")
rep_ds = load_trimmed_dataset(True, tf.uint8) # trainset
print("Finshed loading dataset")
def representative_data_gen():
  for input_value in rep_ds.batch(1).take(500):
    # Model has only one input so each data point has one element.
    yield [input_value]

if not os.path.exists(int_quant_save_to):
    os.mkdir(int_quant_save_to)


kmodels = [tf.keras.models.load_model(os.path.join(path_to, file)) for file in model_list]
existing_tflite_model = None

class ReshapeOp(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[32,32], dtype=tf.float32)])
    def __call__(self,x):
        s1 = tf.expand_dims(x,2)
        s1 = tf.expand_dims(s1,0)
        return existing_tflite_model(s1)


class AltCNNOp(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[32,32], dtype=tf.uint8)])
    def __call__(self,x):
        s1 = tf.expand_dims(x,2)
        s1 = tf.expand_dims(s1,0)
        # s1 = tf.cast(s1, dtype=tf.float32)
        return existing_tflite_model(s1)

for model, model_path in zip(kmodels, model_list):
    existing_tflite_model = model
    hybrid_model = AltCNNOp()
    concrete_func = hybrid_model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = representative_data_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # tf.lite.OpsSet.TFLITE_BUILTINS
    # converter.inference_input_type = tf.float32
    # converter.inference_output_type = tf.float32
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    destination = os.path.join(int_quant_save_to, model_path.split("/")[1] + ".tflite")
    with open(destination, 'wb') as f:
        f.write(tflite_quant_model)
    path = "/mnt/c/Users/Aeyohan/Documents/work/ENGG4811/code/Tensorflow/"
    command = 'ubuntu run "xxd -i ' + path + destination + ' > ' + path + destination.split(".tflite")[0]+".cpp" + '"' 
    command = command.replace("\\", "/")
    print(command)
    os.system(command)
        

# %%


def load_trimmed_dataset(binary, type=tf.float32):
    
    cacheFile = open("tf_spec_graph_cache.pic", 'rb')
    data = pickle.load(cacheFile)

    files = data[0]
    spec_graphs = data[1]
    graphs_dim = [tf.expand_dims(value, 2) for value in spec_graphs]
    graphs_dim = [tf.cast(value, tf.float32) for value in graphs_dim]
    graphs_dim = [(value-133) / 2.1 for value in graphs_dim]
    graphs_int16 = [tf.cast(value, type) for value in graphs_dim]
    
    # load labels
    raininglabelList = {"Yes": 1, "No": 0}
    typelabelList = {"None": 0, "Very Light": 1, "Light": 2, "Moderate": 3, "Heavy": 4, "Very Heavy": 5}
    labelFile = pd.read_csv("E:\\Data\\Thesis\\Audio\\filteredOutputs\\labels.csv")
    folderList = labelFile["Folder"]
    sampleList = labelFile["Sample"]
    pathList = []
    for i in range(len(folderList)):
        pathList.append(folderList + "\\" + sampleList)
    rainLabels = labelFile["Raining"].apply(lambda rainLabel: raininglabelList[rainLabel])
    typeLabels = labelFile["Rain type"].apply(lambda typeLabel: typelabelList[typeLabel])

    rainLabels = rainLabels.tolist()
    typeLabels = typeLabels.tolist()

    rainLabelsSliced = []
    typeLabelsSliced = []

    for value in rainLabels:
        rainLabelsSliced.extend([value,value,value,value,value])

    for value in typeLabels:
        typeLabelsSliced.extend([value,value,value,value,value]) 

    rainLabelsSliced = [tf.cast(value,tf.int16) for value in rainLabelsSliced]
    typeLabelsSliced = [tf.cast(value,tf.int16) for value in typeLabelsSliced]

    selData = None
    if binary:
        selData = rainLabelsSliced
    else:
        selData = typeLabelsSliced
    # graphs_int16 = tf.convert_to_tensor(graphs_int16)
    # selData = tf.convert_to_tensor(selData)
    # selData = tf.reshape(selData, [len(rainLabelsSliced), 1,1,1])
    # typeLabelsSliced = tf.convert_to_tensor(typeLabelsSliced)
    # preprocess stfts as necesasry
    ft_ds = tf.data.Dataset.from_tensor_slices(graphs_int16)
    ft_ds.element_spec

    # cachedSet = ft_ds.cache()
    shuffle = ft_ds.shuffle(8000)
    return shuffle 
    


# %% get the name of output files and network_len of each file
if __name__ == "__main__":
    input_path = "binary_int_quant_models"

    filePaths = []
    for root, folders, files in os.walk(input_path):
        for fileName in files:
            if fileName.__contains__(".cpp"):
                filePaths.append(os.path.join(root, fileName))

    results = []
    for fileName in filePaths:
        f = open(fileName, 'r')
        f.readline()
        for line in f:
            if line.__contains__("unsigned"):
                size = line.split("=")[1].split(";")[0]
                results.append((fileName, size))

# %% read given output files and create a copy with standardised variable names

if __name__ == "__main__":
    input_path = "binary_int_quant_models"
    output_path = "binary_int_quant_deploy"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # load files
    filePaths = []
    for root, folders, files in os.walk(input_path):
        for fileName in files:
            if fileName.__contains__(".cpp"):
                filePaths.append(os.path.join(root, fileName))

    # grab the contents of each file and modify the variable names
    contents = []

    for fileName in filePaths:
        f = open(fileName, 'r')
        f.readline() # ignore the first file
        lines = []
        lines.append('#include "specg_small_compound.h"\n')
        lines.append("alignas(8) const unsigned char compound_model_tflite[] = {")
        for line in f:
            value = line
            if value.__contains__("unsigned"):
                size = line.split("=")[1].split(";")[0]
                value = "const unsigned int audio_tflite_model_len = " + str(size) + ";"
            lines.append(value)
        contents.append(lines)

    # write the files to the output directory
    fileIDs = [path.split(".cpp")[0] for path in filePaths]
    for file, lines in zip(fileIDs, contents):
        output_dir_path = os.path.join(output_path, file.split("\\")[1])
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        f = open(os.path.join(output_dir_path, file.split("\\")[1] + ".cpp"), 'w')
        for line in lines:
            f.write(line)
        f.close()
# %%
